function [ obj_est ] = recon_sparse_rank(filename,y,R0,mu1,mu2,num_irls,num_grade,size1,size2,size3)
%2017.1.17 ����ÿ��ѭ���ع����
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% g_noise/gamma_noise,��������ģ��Ԥ����ʵ�ʹ۲�֮��Ĳ��졣
% ��ͼ���ؽ��У���������ȷ���ؽ���ͼ�񾡿��ܽӽ��۲����ݡ�
% ͨ������Щ��������������������������������һ��
% ���γ�������Ŀ�꺯������ָ���Ż��㷨�ҵ����Ž⡣
%%%%%reconstruction%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if pt==1;
%     %y��R��Ԥ����
%     y=y-mean(y);%y����ֵ
%     R0=R0-ones(length(y),1)*mean(R0,1);%Rÿ�м�ȥ��ֵ
% end
size12=size1*size2;
%���ó�ʼ����
amin=single(1e-30);
amax=single(1e30);
a0=single(1e-6);
dtol=1e-7;
p=1;
num_iter=0;
mu2=mu2*1;

%���ݹ�һ��


% max_r=max(max(R0));
% R0=R0/max_r;
size_obj=size(R0,2);     % 2,�������������1�����������
%����single�ͳ�ʼ�ռ�  (�У��У�ͨ����)
obj_est=zeros(size_obj,1,'single');


g_tv=zeros(size1,size2,size3,'single');
gg_dp=zeros(size1,size2,size3,'single');
gamma_dp=zeros(size_obj,1,'single');
obj_hdp_mat=zeros(size1,size2,size3,'single');
obj_vdp_mat=zeros(size1,size2,size3,'single');

%���������
obj_mat=reshape(obj_est,size1,size2,size3);
obj_hd=diff(obj_mat,1,2);                           % ��p,m,c���Ծ������е�����Ԫ�ؽ��в�ִ������ؽ����ԭ���Ļ�������һ��
obj_vd=diff(obj_mat,1,1);                           % ��p,m,c���Ծ������е�����Ԫ�ؽ��в�ִ������ؽ����ԭ���Ļ�������һ��
resid_y=R0*obj_est-y;                               % resid_y=-y����y

%Ҫʹy�������㸽�������ݲ������ã���Ҫ��ͼ���ؽ����� �� resid_y�������һ��������ͼ�����ת���ɵ���������
% �������ͼ��������Ԫ����y������Ӧ��ͼ����ͬ�����ĻҶȷֲ�Ϊ�����������Ӧ��λ�ø�����Ϊ0�����඼Ϊ1.
%����ͼ���������"ZC_y754_x1064.tif"�����ķֲ�
% spc_matrix =single(imread('spc_matrix.tiff'));
% spc_matrix=reshape(spc_matrix',160*160,1);     %spc_matrix'�����ת��'
% resid_y=resid_y.*spc_matrix;



%IRLS�㷨
for loopk=1:num_irls
    
  %�����ݶ������
  obj_hdp=mu1*(abs(obj_hd)+dtol).^(p-2);
  obj_vdp=mu1*(abs(obj_vd)+dtol).^(p-2);
          
  %����ֵ�ֽ�
  obj_x2=reshape(obj_est,size1*size2,size3);
  [u,~,v]=svd(obj_x2, 0); 
  re_v=reshape(repmat(v',size12,1),size3,size_obj)';
  u_v=re_v.*repmat(u(:,1:size3),size3,1);
  g_rank=0.5*mu2*sum(u_v,2);     %ʵ�ֶ�����ֵ��L1�����ļ��㡣�˷����� g_rank(����Լ��) �������ݶ��½������У���Ϊ�����
   
  %�ݶ��㷨
  for loopl=1:num_grade

   %������˷�������ݶ�
   g_noise=R0'*resid_y;   %��ֵ�ر����Ϊʲô�أ�δ��y����һ����Ӱ�첻��  python�ж�R0����һ���ķ�������

   g_htv=obj_hdp.*obj_hd;
   g_vtv=obj_vdp.*obj_vd;   
   g_tv(:,1:(size2-1),:)=-g_htv;
   g_tv(:,2:size2,:)= g_tv(:,2:size2,:)+g_htv;
   g_tv(1:(size1-1),:,:)=g_tv(1:(size1-1),:,:)-g_vtv;
   g_tv(2:size1,:,:)=g_tv(2:size1,:,:)+g_vtv;
        
   g_obj=2*(g_noise+reshape(g_tv,size_obj,1)+g_rank);  %g_tv,��ˮƽ�ʹ�ֱ�������ۻ��ݶȵ�������ͨ���ۻ���Ȩ�ݶȵ�ƽ��������һ���������������������ͼ����ܱ��
   %g_obj��ʾ�ݶ�����
   %��������ݶ����gamma  �����ڵ���������ȷ���㷨���ȶ��Ժ������ԡ���Ҳ���ܻ������㷨�ļ��㸴�Ӷ�
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

   %�������alpha
   if gamma<=0
       alpha=a0;
   else
       alpha=(g_obj'*g_obj)/gamma;
       alpha=max(min(alpha,amax),amin);
   end
     
   %��������-----------------�иĶ�
   obj_est=max(obj_est-g_obj*alpha,0);  %ϡ��Լ��
  % ���µ������̱��沿���Լ���ȥ 
   if mod(loopk*loopl,20)==0
   imwrite(uint8(reshape(obj_est,size1,size2*size3)/max(obj_est)*255),strcat(filename,'_iter_',num2str(loopk*loopl),'.tif'));
   end
   
   if mod(loopk*loopl,num_irls*num_grade)==0
   imwrite(uint8(reshape(obj_est,size1,size2*size3)/max(obj_est)*255),strcat(filename,'_iter_',num2str(loopk*loopl),'.tif'));
   end
   %�����´��õ��Ĳ��
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

