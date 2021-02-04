# Rossmann Dataset Analysis DSR Mini-Competition

## Project Description

The aim of this project is to apply a prediction model to sales in Rossmann stores according provided data. Rossmann is one of the largest drug store chains in Europe established in 1972. 

## Data Set

`train.csv`- training data for sales

`holdout.csv`- test set

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
Python 3.7 or later and the following packages: 

`pandas`
`numpy`
`scikit-learn`
`matplotlib`
`seaborn`
`xgboost`
`holidays`

## How to use this repository
Clone the repository to your local machine by running:

`git clone https://github.com/wrijupan/Rossmann-DSR-minicomp`

Create a python environment (3.7 or later) and install the requirements from file

`requirements.txt`. 

Run the notebook 

`rossmann_sales_predict.ipynb` 

that is found in the results directory using this environment.