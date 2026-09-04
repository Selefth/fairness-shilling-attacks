This dataset includes long-term (about 18 months from April 2012 to September 2013) global-scale check-in data collected from Foursquare.
It contains 33,278,683 checkins by 266,909 users on 3,680,126 venues (in 415 cities in 77 countries). Those 415 cities are the most checked 415 cities in the world, each of which contains at least 10000 check-ins). Please see the references for more details about data collection and processing.
It contains three files in tsv format.

- File dataset_TIST2015_Checkins.txt contains all check-ins with 4 columns, which are:
1. User ID (anonymized)
2. Venue ID (Foursquare)
3. UTC time
4. Timezone offset in minutes (The offset in minutes between when this check-sin occurred and the same time in UTC, i.e., UTC time + offset is the local time)

- File dataset_TIST2015_POIs.txt contains all venue data with 7 columns, which are:
1. Venue ID (Foursquare) 
2. Latitude
3. Longitude
4. Venue category name (Foursquare)
5. Country code (ISO 3166-1 alpha-2 two-letter country codes)

- File dataset_TIST2015_Cities.txt contains all 415 cities data with 6 columns, which are:
Venue category ID (Foursquare)
1. City name
2. Latitude (of City center)
3. Longitude (of City center)
4. Country code (ISO 3166-1 alpha-2 two-letter country codes)
5. Country name
6. City type (e.g., national capital, provincial capital)


=============================================================================================================================
Please cite our papers if you publish material based on this dataset.

=============================================================================================================================
REFERENCES

@article{yang2015nationtelescope,
  title={NationTelescope: Monitoring and visualizing large-scale collective behavior in LBSNs},
  author={Yang, Dingqi and Zhang, Daqing and Chen, Longbiao and Qu, Bingqing},
  journal={Journal of Network and Computer Applications},
  volume={55},
  pages={170--180},
  year={2015},
  publisher={Elsevier}
}

@article{yang2015participatory,
  title={Participatory cultural mapping based on collective behavior in location based social networks},
  author={Yang, Dingqi and Zhang, Daqing and Qu, Bingqing},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2015},
  note = {in press},
  publisher={ACM}
}
