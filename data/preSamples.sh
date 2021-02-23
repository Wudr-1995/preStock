j=1
for (( i=1; i<=2225; i++ ))
do
	cat ./Input.txt | tail -n +$j | head -n 10 >> ./Samples/Day10Input${i}.txt
	tmp=$(cat ./Input.txt | tail -n +`expr $j + 19` | head -n 1)
	tmp=$(echo $tmp | awk '{print $2}')
	echo $tmp >> ./Samples/Day10Label${i}.txt
	j=`expr $j + 1`
done
