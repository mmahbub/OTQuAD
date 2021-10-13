# OTQuAD
Opioid Therapy Question Answering Dataset

This repo contains the question answering dataset on VA/DoD clinical practice guideline for opioid therapy for chronic pain

The resources used to make this dataset are:

  1. https://www.pbm.va.gov/AcademicDetailingService/Documents/Pain_Opioid_Taper_Tool_IB_10_939_P96820.pdf
  2. https://www.healthquality.va.gov/guidelines/Pain/cot/VADoDOTCPG022717.pdf

Currently there are 249 QA pairs and 23 contexts created from resource 1.

Creating more QA pairs and contexts using resource 2 is work-in-progress ...


Current Data Stats:

  * SQuAD has 1204 contexts and 11873 QA pairs in the dev dataset. The ratio of `QA pairs to contexts` is ~10
  * As of now, the ratio of `QA pairs to contexts` in OTQuAD is ~11.
