import os.path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from lightGCN import LightGCN
from utils import BPRLoss, EmbLoss 


class MBGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(MBGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index 
        self.all_edge_index = dataset.all_edge_index 
        self.behaviors = args.behaviors 
        self.embedding_size = args.embedding_size 

        self.args = args
        self.reg_coeff = args.reg_coeff 
        self.main_coeff = args.main_coeff 
         

        
        # 1. ID 嵌入
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1) #
        
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.1) #

        # 2. “行为特定 GCN” 
        self.behavior_gcns = nn.ModuleDict({
            behavior: LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, inter, self.embedding_size)
            for behavior, inter in dataset.inter_matrix.items()
        }) #

        # 3. “行为(协同)视图 GCN”  全行为合并
        self.behavioral_gcn = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, 
                                       dataset.all_inter_matrix, self.embedding_size)

        # 4. 频谱滤波器 
        if self.embedding_size % 2 != 0:
            raise ValueError("Embedding size must be even for torch.fft.rfft.")
        
        # 为每种行为创建一个可训练的“去噪”滤波器
      
        self.behavior_complex_weights = nn.Parameter(
            torch.randn(len(self.behaviors), self.embedding_size // 2 + 1, 2, dtype=torch.float32)
        )

        # 全局滤波器
        self.global_complex_weight = nn.Parameter(torch.randn(1, self.embedding_size // 2 + 1, 2, dtype=torch.float32))

       # 5. 模态感知偏好  模块 
        
        self.query_w1 = nn.Parameter(torch.randn(len(self.behaviors), self.embedding_size, self.embedding_size))
        self.query_b1 = nn.Parameter(torch.zeros(len(self.behaviors), self.embedding_size))
        
        self.query_w2 = nn.Parameter(torch.randn(len(self.behaviors), self.embedding_size, self.embedding_size))
      
        nn.init.xavier_uniform_(self.query_w1)
        nn.init.xavier_uniform_(self.query_w2)

        self.softmax = nn.Softmax(dim=-1)

       
        self.dropout = nn.Dropout(p=args.dropout_rate if 'dropout_rate' in args else 0.1)
        # self.behavior_weights = nn.Parameter(torch.ones(len(self.behaviors))/len(self.behaviors))#zhu
        # self.dropout = nn.Dropout(p=args.dropout_rate if 'dropout_rate'in args else 0.1)

        # ---  ---

        self.bpr_loss = BPRLoss() #
        self.emb_loss = EmbLoss() #

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None
        self.storage_side_user_embeddings = None
        self.storage_side_item_embeddings = None
        self.storage_content_user_embeddings = None
        self.storage_content_item_embeddings = None

        self._load_model()

    def invalidate_cache(self):
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None
        self.storage_side_user_embeddings = None
        self.storage_content_user_embeddings = None
        self.storage_side_item_embeddings = None
        self.storage_content_item_embeddings = None


    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False) 

    
    def spectrum_behavior_convolution(self, local_embeddings_stack, global_embedding):#频谱行为卷积，输入来自各个行为的GCN局部嵌入堆叠，来自协同视图GCN的全局嵌入

        batch_size, n_behaviors, emb_dim = local_embeddings_stack.shape
        
        #对所有行为fft
        local_fft = torch.fft.rfft(local_embeddings_stack, dim=2, norm='ortho')
        filter_weight = torch.view_as_complex(self.behavior_complex_weights)
        #滤波
        denoised_fft = local_fft * filter_weight.unsqueeze(0)
        denoised_embeds_stack = torch.fft.irfft(denoised_fft, n=emb_dim, dim=2, norm='ortho')
        
        #全局处理
        global_fft = torch.fft.rfft(global_embedding, dim=1, norm='ortho')
        global_filter_weight = torch.view_as_complex(self.global_complex_weight)
        processed_global_fft = global_fft * global_filter_weight
        processed_global = torch.fft.irfft(processed_global_fft, n=emb_dim, dim=1, norm='ortho')
        

        return denoised_embeds_stack, processed_global #输出去噪后的局部和整体嵌入

    def apply_fusion(self, local_embeddings_stack, content_embeddings):#注意力机制融合
        #1.频谱处理
        denoised_stack, processed_global = self.spectrum_behavior_convolution(local_embeddings_stack, content_embeddings)
        
        # last_behavior_emb = local_embeddings_stack[:, -1, :]  # shape: [batch_size, emb_dim]
        # mean_agg = torch.mean(denoised_stack, dim=1)  # 对行为维度求平均
        # side_embeddings = mean_agg + processed_global

        #2.注意力计算
         #查询向量
        h = torch.einsum('bi, nij -> bnj', processed_global, self.query_w1)
        h = h + self.query_b1.unsqueeze(0) 
        h = torch.tanh(h)
         #计算注意力分数
        attn_scores = torch.einsum('bnj, njk -> bnk', h, self.query_w2)
        att_weights = self.softmax(attn_scores) 
        
        #3.加权融合
        weighted_locals = att_weights * denoised_stack 
        mean_agg = torch.mean(weighted_locals, dim=1) 
        
        #4.特征融合
        side_embeddings = mean_agg * self.args.mean_agg_coeff + processed_global * self.args.processed_global_coeff
        final_embeddings =  side_embeddings + content_embeddings * self.args.content_coeff
        
        
        return final_embeddings, side_embeddings


    def gcn_propagate(self, user_id_emb, item_id_emb):#GCN传播 
    
        user_embeddings, item_embeddings = [], []
        # 1. 行为特定 GCN
        for behavior in self.behaviors:
            behavior_embeddings = self.behavior_gcns[behavior](user_id_emb, item_id_emb) #
            user_embedding, item_embedding = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1]) #
            
            user_embeddings.append(user_embedding)
            item_embeddings.append(item_embedding) #

        # (N_users, N_behaviors, Emb_dim)
        all_user_embeddings = torch.stack(user_embeddings, dim=1) #
        all_item_embeddings = torch.stack(item_embeddings, dim=1) #

        # 2. 协同(内容)视图 GCN
        content_embeddings = self.behavioral_gcn(user_id_emb, item_id_emb)
        content_user_embeds, content_item_embeds = torch.split(content_embeddings, [self.n_users + 1, self.n_items + 1])

        return all_user_embeddings, all_item_embeddings, content_user_embeds, content_item_embeds

    
    def get_final_embeddings(self):
      

        user_id_preference = self.user_embedding.weight
        item_id_preference = self.item_embedding.weight
        
        # 1. GCN 传播 (获取所有视图的表征)
        all_user_embeds, all_item_embeds, \
        content_user_embeds, content_item_embeds = self.gcn_propagate(user_id_preference, item_id_preference)
    
        final_user_embeds, side_user_embeds = self.apply_fusion(all_user_embeds, content_user_embeds)
        final_item_embeds, side_item_embeds = self.apply_fusion(all_item_embeds, content_item_embeds)
        

        return (final_user_embeds, final_item_embeds, 
                side_user_embeds, content_user_embeds, 
                side_item_embeds, content_item_embeds)

    def forward(self, batch_users, batch_posItems, batch_negItems, epoch=None):
        
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None
    
        # 1. 获取所有最终表征
        # ua, ia 分别是 final_user_embeds 和 final_item_embeds
        # side_u, content_u 是  用的两个视图
        ua, ia, side_u, content_u, side_i, content_i = self.get_final_embeddings()

        # 2. 准备 BPR Loss 的数据
        users = batch_users
        positems = batch_posItems
        negItems = batch_negItems

        # 3. 提取批次表征
        users_final_emb = ua[users]
        positems_final_emb = ia[positems]
        negItems_final_emb = ia[negItems]

        # 4. 计算 BPR Loss 
        posscores = torch.sum(users_final_emb * positems_final_emb, dim=1)
        negscores = torch.sum(users_final_emb * negItems_final_emb, dim=1)
        main_loss = self.bpr_loss(posscores, negscores)  * self.main_coeff

        # 6. 计算正则化损失 (Reg Loss)
        reg_loss = self.emb_loss(users_final_emb, positems_final_emb, negItems_final_emb)
        reg_loss = self.reg_coeff * reg_loss

        # 打印损失 
        print(f"loss: {main_loss.item(), reg_loss.item()}", flush=True)

        # 7. 总损失 
        total_loss = main_loss + reg_loss 

        return total_loss

    def full_predict(self, users):
        
        if self.storage_user_embeddings is None:
            # 获取最终的、融合后的表征
            final_user_embs, final_item_embs, _, _, _, _ = self.get_final_embeddings()
            print(f"11", flush=True)
 
            self.storage_user_embeddings = final_user_embs
            self.storage_item_embeddings = final_item_embs
    

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores

 