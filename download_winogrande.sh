#!/bin/sh

wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
unzip winogrande_1.1.zip; rm -rf __MACOSX
mv winogrande_1.1 data
rm winogrande_1.1.zip

