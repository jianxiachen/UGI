function [ obj_est ] = recon_sparse_rank(filename,y,R0,mu1,mu2,num_irls,num_grade,size1,size2,size3)
%2017.1.17 保存每次循环重构结果
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
% g_noise/gamma_noise,用于量化模型预测与实际观测之间的差异。
% 在图像重建中，这有助于确保重建的图像尽可能接近观测数据。
% 通常，这些项会与其他项（如数据拟合项和正则化项）结合在一起，
% 以形成完整的目标函数，并指导优化算法找到最优解。
%%%%%reconstruction%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if pt==1;
%     %y和R的预处理
%     y=y-mean(y);%y减均值
%     R0=R0-ones(length(y),1)*mean(R0,1);%R每列减去均值
% end
size12=size1*size2;
%设置初始参数
amin=single(1e-30);
amax=single(1e30);
a0=single(1e-6);
dtol=1e-7;
p=1;
num_iter=0;
mu2=mu2*1;

%数据归一化


% max_r=max(max(R0));
% R0=R0/max_r;
size_obj=size(R0,2);     % 2,计算矩阵列数；1计算矩阵行数
%设置single型初始空间  (行，列，通道数)
obj_est=zeros(size_obj,1,'single');


g_tv=zeros(size1,size2,size3,'single');
gg_dp=zeros(size1,size2,size3,'single');
gamma_dp=zeros(size_obj,1,'single');
obj_hdp_mat=zeros(size1,size2,size3,'single');
obj_vdp_mat=zeros(size1,size2,size3,'single');

%计算相关项
obj_mat=reshape(obj_est,size1,size2,size3);
obj_hd=diff(obj_mat,1,2);                           % （p,m,c）对矩阵行中的连续元素进行差分处理，返回结果在原来的基础上少一行
obj_vd=diff(obj_mat,1,1);                           % （p,m,c）对矩阵列中的连续元素进行差分处理，返回结果在原来的基础上少一列
resid_y=R0*obj_est-y;                               % resid_y=-y或者y

%要使y中心亮点附近的数据不起作用，需要将图像重建程序 中 resid_y变量点乘一个由特殊图像矩阵转换成的列向量。
% 这个特殊图像矩阵的像元数和y向量对应的图像相同。它的灰度分布为：中心亮点对应的位置附近都为0，其余都为1.
%特殊图像矩阵类似"ZC_y754_x1064.tif"这样的分布
% spc_matrix =single(imread('spc_matrix.tiff'));
% spc_matrix=reshape(spc_matrix',160*160,1);     %spc_matrix'必须加转置'
% resid_y=resid_y.*spc_matrix;



%IRLS算法
for loopk=1:num_irls
    
  %计算梯度相关项
  obj_hdp=mu1*(abs(obj_hd)+dtol).^(p-2);
  obj_vdp=mu1*(abs(obj_vd)+dtol).^(p-2);
          
  %奇异值分解
  obj_x2=reshape(obj_est,size1*size2,size3);
  [u,~,v]=svd(obj_x2, 0); 
  re_v=reshape(repmat(v',size12,1),size3,size_obj)';
  u_v=re_v.*repmat(u(:,1:size3),size3,1);
  g_rank=0.5*mu2*sum(u_v,2);     %实现对奇异值的L1范数的计算。核范数项 g_rank(低秩约束) 被纳入梯度下降步骤中，作为正则化项。
   
  %梯度算法
  for loopl=1:num_grade

   %计算除核范数外的梯度
   g_noise=R0'*resid_y;   %数值特别大，是为什么呢？未对y作归一化（影响不大）  python中对R0做归一化的方法对吗？

   g_htv=obj_hdp.*obj_hd;
   g_vtv=obj_vdp.*obj_vd;   
   g_tv(:,1:(size2-1),:)=-g_htv;
   g_tv(:,2:size2,:)= g_tv(:,2:size2,:)+g_htv;
   g_tv(1:(size1-1),:,:)=g_tv(1:(size1-1),:,:)-g_vtv;
   g_tv(2:size1,:,:)=g_tv(2:size1,:,:)+g_vtv;
        
   g_obj=2*(g_noise+reshape(g_tv,size_obj,1)+g_rank);  %g_tv,在水平和垂直方向上累积梯度的能量。通过累积加权梯度的平方来构建一个能量项，这个能量项代表了图像的总变差
   %g_obj表示梯度向量
   %计算二次梯度相关gamma  ：用于调整步长，确保算法的稳定性和收敛性。但也可能会增加算法的计算复杂度
   speckle_g=R0*g_obj;  
   gamma_noise=speckle_g'*speckle_g;
        
   gg_dp(:,1:(size2-1),:)=obj_hdp;
   gg_dp(:,2:size2,:)=gg_dp(:,2:size2,:)+obj_hdp;
   gg_dp(1:(size1-1),:,:)=gg_dp(1:(size1-1),:,:)+obj_vdp;
   gg_dp(2:size1,:,:)=gg_dp(2:size1,:,:)+obj_vdp;
   gg_dp_re=reshape(gg_dp,size_obj,1);
        
   obj_hdp_mat(:,1:(size2-1),:)=obj_hdp;
   obj_hdp_re=reshape(obj_hdp_mat,size_obj,1);
   obj_vdp_mat(1:(size1-1),:,:)=obj_vdp;
   obj_vdp_re=reshape(obj_vdp_mat,size_obj,1);
   gamma_dp(1:(size_obj-size1))=-g_obj((size1+1):size_obj).*obj_hdp_re(1:(size_obj-size1));
   gamma_dp((size1+1):size_obj)=gamma_dp((size1+1):size_obj)-g_obj(1:(size_obj-size1)).*obj_hdp_re(1:(size_obj-size1));
   gamma_dp(1:(size_obj-1))=gamma_dp(1:(size_obj-1))-g_obj(2:size_obj).*obj_vdp_re(1:(size_obj-1));
   gamma_dp(2:size_obj)=gamma_dp(2:size_obj)-g_obj(1:(size_obj-1)).*obj_vdp_re(1:(size_obj-1));
           
   gamma=(gamma_noise+g_obj'*(gg_dp_re.*g_obj+gamma_dp))*2;

   %计算参数alpha
   if gamma<=0
       alpha=a0;
   else
       alpha=(g_obj'*g_obj)/gamma;
       alpha=max(min(alpha,amax),amin);
   end
     
   %更新物体-----------------有改动
   obj_est=max(obj_est-g_obj*alpha,0);  %稀疏约束
  % 以下迭代过程保存部分自己隐去 
   if mod(loopk*loopl,20)==0
   imwrite(uint8(reshape(obj_est,size1,size2*size3)/max(obj_est)*255),strcat(filename,'_iter_',num2str(loopk*loopl),'.tif'));
   end
   
   if mod(loopk*loopl,num_irls*num_grade)==0
   imwrite(uint8(reshape(obj_est,size1,size2*size3)/max(obj_est)*255),strcat(filename,'_iter_',num2str(loopk*loopl),'.tif'));
   end
   %计算下次用到的差分
   obj_mat=reshape(obj_est,size1,size2,size3);
   obj_hd=diff(obj_mat,1,2);
   obj_vd=diff(obj_mat,1,1);  
   resid_y=R0*obj_est-y;
   
%    spc_matrix =single(imread('spc_matrix.tiff'));
%    spc_matrix=reshape(spc_matrix',160*160,1);
%    resid_y=resid_y.*spc_matrix;
     

   num_iter=num_iter+1;
%    percent=50+num_iter/num_iter_g/num_iter_irls*100*0.5;
%    fid=fopen(filename_proc_info,'wt');
%    fprintf(fid,'[reconstruct]\r\n');
%    fprintf(fid,'Start = 1\r\n');
%    fprintf(fid,'Status = %g\r\n',percent);
%    fclose(fid);
  end
end
% if norm==1
%     obj_est=obj_est/max(obj_est);
% end
end

