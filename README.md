# AutoLearn - Automated Feature Generation and Selection (ICDM 2017)

This repository contains source code for the research work described in:

[AutoLearn — Automated Feature Generation and Selection](https://ieeexplore.ieee.org/abstract/document/8215494)


## Dependencies

* **[Scikit-learn](http://scikit-learn.org/stable/install.html)** for modelling
* **[Pandas](https://pandas.pydata.org/)** for data manipulation
* **[Numpy](http://www.numpy.org/)** for performing mathematical operations
* **[Matplotlib](https://matplotlib.org/)** for plotting 2D graphs

## How to Run
```
python main.py
```


## Note
-- The current version of the code is not optimized. I will updated the optimized version in coming weeks.

-- The hyperparameter value of the thresholds are hard coded. Use hyperparameter optimization for best results.

-- All dataset files should follow the exact same template as that of sonar.csv (class labels column at the last)


## Citation


   If you use our code, please cite the following papers:
    
    @inproceedings{kaul2017autolearn,
    title={AutoLearn—Automated Feature Generation and Selection},
    author={Kaul, Ambika and Maheshwary, Saket and Pudi, Vikram},
    booktitle={Data Mining (ICDM), 2017 IEEE International Conference on},
    pages={217--226},
    year={2017},
    organization={IEEE}
    }   
    

    @MISC{saket:automl_pdf,
      AUTHOR = "Maheshwary, Saket and Kaul, Ambika and Pudi, Vikram",
      TITLE = "Data Driven Feature Learning",
      MONTH = Aug,
      YEAR = 2017,
      NOTE =    "\url{https://www.researchgate.net/profile/Saket_Maheshwary/publication/325736313_Data_Driven_Feature_Learning/links/5b20e25ca6fdcc69745d796c/Data-Driven-Feature-Learning.pdf}"
    }

