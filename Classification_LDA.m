function accuracy=Classification_LDA(document_path)
 
p = genpath(document_path);
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
 
test=data33;
test_group=label3;
 
[model,k,ClassLabel]=LDATraining(train,train_group);
target=LDATesting(test,k,model,ClassLabel);
disp('分类结果：');
disp(target);
target=target-test_group;
t=length(find(target==0));
[r,l]=size(test_group);
accuracy=t/r;
disp('accuracy：');
disp(accuracy);

end