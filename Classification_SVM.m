function accuracy=Classification_SVM(document_path)
 
p = genpath(document_path);
%p = genpath('D:\四分类\Cdata');
length_p = size(p,2);%字符串p的长度  
path = {};%建立一个单元数组，数组的每个单元中包含一个目录  
temp = [];  
data22=[];
data33=[];
label=[];
label3=[];
for i = 1:length_p %寻找分割符';'，一旦找到，则将路径temp写入path数组中  
    if p(i) ~= ';'  
        temp = [temp p(i)];  
    else   
        temp = [temp '\']; %在路径的最后加入 '\'  
        path = [path ; temp];  
        temp = [];  
    end  
end    
clear p length_p temp;  
%至此获得data文件夹及其所有子文件夹（及子文件夹的子文件夹）的路径，存于数组path中。  
%下面是逐一文件夹中读取图像  
file_num = size(path,1);% 子文件夹的个数  
k=1;
for i = 1:file_num  
    file_path =  path{i}; % 图像文件夹路径  
    img_path_list = dir(strcat(file_path,'*.mat'));  
    img_num = length(img_path_list); %该文件夹中图像数量  
    if img_num > 0  
        for j = 1:img_num
            image_name = img_path_list(j).name;% 图像名  
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
%全部数据
%AllData=[data22(:,:),label(:,1)];
train=data22;
train_group=label;
%%训练数据处理
[train,pstrain] = mapminmax(train');
% 将映射函数的范围参数分别置为0和1
pstrain.ymin = 0;
pstrain.ymax = 1;
% 对训练集进行[0,1]归一化
[train,pstrain] = mapminmax(train,pstrain);

test=data33;
test_group=label3;
%%测试数据处理
[test,pstest] = mapminmax(test');
% 将映射函数的范围参数分别置为0和1
pstest.ymin = 0;
pstest.ymax = 1;
% 对测试集进行[0,1]归一化
[test,pstest] = mapminmax(test,pstest);

% 对训练集和测试集进行转置,以符合libsvm工具箱的数据格式要求
train = train';
test = test';

%寻找最优c和g
%粗略选择：c&g 的变化范围是 2^(-10),2^(-9),...,2^(10)
%[bestacc,bestc,bestg] = SVMcgForClass(train_group,train,-10,10,-10,10);
[bestacc,bestc,bestg] = SVMcgForClass(train,train_group,-10,10,-10,10);
%精细选择：c 的变化范围是 2^(-2),2^(-1.5),...,2^(4), g 的变化范围是 2^(-4),2^(-3.5),...,2^(4)
[bestacc,bestc,bestg] = SVMcgForClass(train_group,train,-2,4,-4,4,3,0.5,0.5,0.9);

%训练模型
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model=svmtrain(train_group,train,cmd);
disp(cmd);

%测试分类
[predict_label, accuracy, dec_values]=svmpredict(test_group,test,model);

%打印测试分类结果
figure;
hold on;
plot(test_group,'o');
plot(predict_label,'r*');
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',10);

end
