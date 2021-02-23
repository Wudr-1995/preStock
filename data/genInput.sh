trainL=./trainList.txt
testL=./testList.txt

while read line
do
	tmp=$(echo $line | awk '{print $1}')
	cat $tmp >> samTrainInput.txt
	tmp=$(echo $line | awk '{print $2}')
	cat $tmp >> samTrainLabel.txt
done < $trainL

while read line
do
	tmp=$(echo $line | awk '{print $1}')
	cat $tmp >> samTestInput.txt
	tmp=$(echo $line | awk '{print $2}')
	cat $tmp >> samTestLabel.txt
done < $testL
