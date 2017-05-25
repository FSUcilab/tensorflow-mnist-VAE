mkdir $1
cp *py $1/
cp *x  $1/
cp READ* $1/

(cd $1; ln -s ../data .)
