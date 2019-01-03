function accuracy=Classification_SVM(document_path)
 
p = genpath(document_path);
%p = genpath('D:\�ķ���\Cdata');
length_p = size(p,2);%�ַ���p�ĳ���  
path = {};%����һ����Ԫ���飬�����ÿ����Ԫ�а���һ��Ŀ¼  
temp = [];  
data22=[];
data33=[];
label=[];
label3=[];
for i = 1:length_p %Ѱ�ҷָ��';'��һ���ҵ�����·��tempд��path������  
    if p(i) ~= ';'  
        temp = [temp p(i)];  
    else   
        temp = [temp '\']; %��·���������� '\'  
        path = [path ; temp];  
        temp = [];  
    end  
end    
clear p length_p temp;  
%���˻��data�ļ��м����������ļ��У������ļ��е����ļ��У���·������������path�С�  
%��������һ�ļ����ж�ȡͼ��  
file_num = size(path,1);% ���ļ��еĸ���  
k=1;
for i = 1:file_num  
    file_path =  path{i}; % ͼ���ļ���·��  
    img_path_list = dir(strcat(file_path,'*.mat'));  
    img_num = length(img_path_list); %���ļ�����ͼ������  
    if img_num > 0  
        for j = 1:img_num
            image_name = img_path_list(j).name;% ͼ����  
            name = strcat(file_path,image_name); 
            data = load(name);
            data1 =data.td;
            [r,c]=size(data1);
            %data2=data1(1:r-15,:);
            data2=data1(1:135,:);
            data22=[data22;data2];
            data3=data1(r-14:r,:);
            data33=[data33;data3];
            %label1=ones(r-15,1)*k;
            label1=ones(135,1)*k;
            label=[label;label1];    
            label2=ones(15,1)*k;
            label3=[label3;label2];
             k=k+1;
        end
    end
   
end
%ȫ������
%AllData=[data22(:,:),label(:,1)];
train=data22;
train_group=label;
%%ѵ�����ݴ���
[train,pstrain] = mapminmax(train');
% ��ӳ�亯���ķ�Χ�����ֱ���Ϊ0��1
pstrain.ymin = 0;
pstrain.ymax = 1;
% ��ѵ��������[0,1]��һ��
[train,pstrain] = mapminmax(train,pstrain);

test=data33;
test_group=label3;
%%�������ݴ���
[test,pstest] = mapminmax(test');
% ��ӳ�亯���ķ�Χ�����ֱ���Ϊ0��1
pstest.ymin = 0;
pstest.ymax = 1;
% �Բ��Լ�����[0,1]��һ��
[test,pstest] = mapminmax(test,pstest);

% ��ѵ�����Ͳ��Լ�����ת��,�Է���libsvm����������ݸ�ʽҪ��
train = train';
test = test';

%Ѱ������c��g
%����ѡ��c&g �ı仯��Χ�� 2^(-10),2^(-9),...,2^(10)
%[bestacc,bestc,bestg] = SVMcgForClass(train_group,train,-10,10,-10,10);
[bestacc,bestc,bestg] = SVMcgForClass(train,train_group,-10,10,-10,10);
%��ϸѡ��c �ı仯��Χ�� 2^(-2),2^(-1.5),...,2^(4), g �ı仯��Χ�� 2^(-4),2^(-3.5),...,2^(4)
[bestacc,bestc,bestg] = SVMcgForClass(train_group,train,-2,4,-4,4,3,0.5,0.5,0.9);

%ѵ��ģ��
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model=svmtrain(train_group,train,cmd);
disp(cmd);

%���Է���
[predict_label, accuracy, dec_values]=svmpredict(test_group,test,model);

%��ӡ���Է�����
figure;
hold on;
plot(test_group,'o');
plot(predict_label,'r*');
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',10);

end
