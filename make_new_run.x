mkdir $1
mv output $1
cp *py $1/
cp -R normalizing_flow $1/
cp *x  $1/
cp READ* $1/
cp -R results $1
rm -rf results

(cd $1; ln -s ../data .)
