result_container = [];

i = 1

while i <= 100
    filepath = strcat('./A/a', int2str(i) ,'.jpg'); 
    image = imread(filepath);
    convertBW = im2bw(image);
    inverter = imcomplement(convertBW);
    
    singleMatrix = inverter(:);
    horizontalData = singleMatrix';
    
    result_container = [result_container; horizontalData];
    i = i + 1
end

csvwriteA('A_binary.txt',result_container)