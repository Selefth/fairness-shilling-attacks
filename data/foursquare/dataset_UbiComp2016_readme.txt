This dataset includes some user profile data for privacy study. It contains 18,201 and 11,874 users who have checked in New York City and Tokyo, respectively. The corresponding user check-in data can be found in the global-scale check-in dataset I published. The two dataset can be linked by the anonymized user ID (the unique key). This data set is collected at the end of Sep. 2013. It contains one tsv file for each city. Please see the references for more details about data collection and processing. 

- Each file dataset_UbiComp2016_UserProfile_{CITY}.txt contains 4 columns, which are:
1. User ID (anonymized but can be linked to the user ID in the global-scale check-in dataset)
2. Gender
3. Twitter friend count
4. Twitter follower count 

=============================================================================================================================
Please cite our papers if you publish material based on this dataset.

=============================================================================================================================
REFERENCES (Bibtex)

@inproceedings{yang2016privcheck,
  title={PrivCheck: privacy-preserving check-in data publishing for personalized location based services},
  author={Yang, Dingqi and Zhang, Daqing and Qu, Bingqing and Cudre-Mauroux, Philippe},
  booktitle={Proceedings of the 2016 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
  pages={545--556},
  year={2016},
  organization={ACM}
}
