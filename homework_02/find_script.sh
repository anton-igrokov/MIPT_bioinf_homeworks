#!/bin/bash

if find . -maxdepth 1 -type f -name "super_secret_key.txt" >/dev/null 2>&1; then
    echo "Found it!" > found.log
else
    echo "I did my best"
fi
# те выводим и ошибки и стдэрр в дев нулл (Все, что вы запишете в /dev/null, будет отброшено, забыто в пустоте. В системе UNIX он известен как нулевое устройство.) + иф
