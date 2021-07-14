name=汽车销售

split -l 80000 $name"labeled.txt"
mv xaa $name"train.txt"
mv xab $name"else.txt"
split -l 10000 $name"else.txt"
mv xaa $name"eval.txt"
mv xab $name"test.txt"
