# MTransE-tf

Constructing and archive for a tensorflow-version MTransE that has been used in the paper below. This version is based on entity-level alignment, instead of triple-level alignment.

	@inproceedings{CTCSZ18,
		author = {Muhao Chen and Yingtao Tian and Kai-Wei Chang and Steven Skiena and Carlo Zaniolo},
		title = {Co-training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment}, 
		booktitle = {IJCAI}, 
		year = {2018},
	}

The wk3l-60k dataset is in the [preprocess](https://github.com/muhaochen/MTransE-tf/tree/master/preprocess) folder.  
Word embeddings and entity list used by KDCoE for wk3l-60k are at [here](http://yellowstone.cs.ucla.edu/~muhao/kdcoe/wk3l_60k_word_embeddings.zip).

### KDCoE
Run training_model2.py to train one turn of the structure encoder. Run test_detail_model2.py to get the threshold (pct in test_try.py) from detail_recall_m2.txt. Run test_try.py to propose new seed alignment from the structure encoder. Run gru_train.py and gru_test.py to set the threshold for the document encoder similarly, and run gru_try.py to propose new seed alignment. Please change input file names for different iterations.

