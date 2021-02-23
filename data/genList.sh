s=Input
for i in ./Samples/*
do
	if [[ $i =~ $s ]]
	then
		label=./Samples/Day10Label${i##*$s}
		echo "${i} ${label}" >> ./totList
	fi
done

gshuf ./totList -o ./randList.txt
