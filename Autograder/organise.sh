#! /bin/bash


mkdir organised


for roll in `sort -n mock_grading/roll_list`
    do
       mkdir organised/"$roll"
       cd organised/"$roll"

        for file in `ls ../../mock_grading/submissions`
            do
                if [[ "$file" == "$roll"* ]]; then

                    ln -s  ../../mock_grading/submissions/"$file"   "$file"

                fi
            done

        cd ../..  
        
    done

exit 0    
