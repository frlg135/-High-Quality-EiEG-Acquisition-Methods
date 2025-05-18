**Data**

* `Data/Origin_Data`: Raw EEG data of Sheep-077
* `Data/Feature_data`: Time- and frequency-domain features extracted from Sheep-077's EEG data
* `Data/Time_Useful_Data`: Power spectral density (PSD) data of Sheep-077

**Code**

* **Time-domain feature extraction**

  * Sample entropy: `Code/Time Feature Analysis/SampEn/SampEn_Origin.py`
  * Other time-domain features: `Code/Time Feature Analysis/OtherFeature/OtherFeal_Origin.py`

* **Frequency-domain feature extraction**

  * All frequency features are computed from the raw data: `Code/Frequency Feature Analysis/Feature.py`

* **Clustering analysis**

  * Performed on extracted features: `Code/Clustering Analysis/K_means_Feature_PCA&t-SNE_New_1Sheep_Improved.py`

* **Correlation analysis**

  * Based on extracted features:

    * Pearson correlation: `Code/Correlation Analysis/Pearson_Feature.py`
    * Spearman rank correlation: `Code/Correlation Analysis/Spearman Rank_Feature.py`
    * Mutual information: `Code/Correlation Analysis/Mutual Information_Feature.py`
    * Dynamic time warping: `Code/Correlation Analysis/Dynamic Time Warping_Feature.py`
    * Correlation matrix visualization: `Code/Correlation Analysis/Correlation Matrix_Feature_English.py`

* **Long-term stability analysis**

  * Based on PSD data: `Code/Long-Term Stability Analysis/Long-Term Stability AnalysisPSD OneSheepEnglish.py`

* **Topological and explainable-AI analysis**

  * Computed from Feature_data:

    * SHAP analysis: `Code/Topological and Explainable-AI Analysis/SHAP_NewOneSheep.py`
    * Mapper topology before SHAP feature selection: `Code/Topological and Explainable-AI Analysis/MapperOneSheep.py`
    * Mapper topology after SHAP feature selection: `Code/Topological and Explainable-AI Analysis/MapperAfterSHAPOneSheep.py`

