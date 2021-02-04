# Rossmann Dataset Analysis DSR Mini-Competition

## Project Description

The aim of this project is to apply a prediction model to sales in Rossmann stores according provided data. Rossmann is one of the largest drug store chains in Europe established in 1972. 

## Getting Started

Prerequisites and the setup is provided in the [samsonafo/DSR_mini_comp](https://github.com/samsonafo/DSR_mini-comp) GitHub repository. 

## Data Set

`train.csv`- training data for sales
`store.csv` - information about each store

### Data Dictionary
```
Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

```

## Requirements

`certifi==2020.12.5
cycler==0.10.0
flake8==3.8.4
holidays==0.10.5.2
importlib-metadata==3.4.0
joblib==1.0.0
kiwisolver==1.3.1
matplotlib==3.3.2
mccabe==0.6.1
numpy==1.20.0
pandas==1.1.3
Pillow==8.1.0
plotly==4.14.3
pycodestyle==2.6.0
pyflakes==2.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
python-dotenv==0.14.0
pytz==2021.1
retrying==1.3.3
scikit-learn==0.23.2
scipy==1.6.0
seaborn==0.11.1
six==1.15.0
threadpoolctl==2.1.0
typing-extensions==3.7.4.3
zipp==3.4.0`


## Data Cleaning and Feature Engineering

## Modeling

## Results



