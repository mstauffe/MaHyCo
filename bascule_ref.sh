#!/bin/bash
#
# This script changes the reference of testcases writted on a list : list_of_cases_to_change
#
function main {
 local readonly curent_dir=$1
 local readonly test_dir=$curent_dir/tests
for cas in $(cat list_of_cases_to_change.txt); do
    local readonly cas_dir=${test_dir}/$cas
    if [[ -d ${cas_dir} ]]; then
	echo CAS=$cas_dir
        echo $curent_dir
	echo $test_dir/$cas
	cp $cas_dir/donnees.txt .
	OMP_NUM_THREADS=3 ./mahyco donnees.txt
        paraview $cas_dir/reference/*.pvd 
	echo "Basculer Yes/No"
	read reponse
	if  [[ "$reponse" == "Yes" ]]; then
	    rm -rf $cas_dir/refrence
	    mv output $cas_dir/refrence
	fi
    fi
done
}

main $@

