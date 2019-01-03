function accuracy=Classification_LDA(document_path)
 
p = genpath(document_path);
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
 
test=data33;
test_group=label3;
 
[model,k,ClassLabel]=LDATraining(train,train_group);
target=LDATesting(test,k,model,ClassLabel);
disp('��������');
disp(target);
target=target-test_group;
t=length(find(target==0));
[r,l]=size(test_group);
accuracy=t/r;
disp('accuracy��');
disp(accuracy);

end