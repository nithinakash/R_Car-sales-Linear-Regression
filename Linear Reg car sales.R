
setwd('C:/Users/Nithin/Downloads/Car sales')

#**************************************************************************************************************#

#########################
#-->Required Packages<--#
#########################
require(dplyr)
require(stringr)
require(fastDummies)
require(ggplot2)
require(caret)
require(car)
require(Metrics)
require(MLmetrics)
require(sqldf)

#**************************************************************************************************************#

################
#-->Datasets<--#
################

car_sales <- read.csv('Car_sales - 1629042283129.csv')

#**************************************************************************************************************#

#################
#-->Data Prep<--#
#################

str(car_sales)

car_sales$Latest_Launch <- as.Date(car_sales$Latest_Launch,format = '%m/%d/%Y')

car_sales$Model <- NULL

#**************************************************************************************************************#

#############
#--> UDF <--#
#############

#cont_var_summary
cont_var_summary <- function(x){
  n = length(x)
  nmiss = sum(is.na(x))
  nmiss_pct = mean(is.na(x))
  sum = sum(x, na.rm=T)
  mean = mean(x, na.rm=T)
  median = quantile(x, p=0.5, na.rm=T)
  std = sd(x, na.rm=T)
  var = var(x, na.rm=T)
  range = max(x, na.rm=T)-min(x, na.rm=T)
  pctl = quantile(x, p=c(0, 0.01, 0.05,0.1,0.25,0.5, 0.75,0.9,0.95,0.99,1), na.rm=T)
  return(c(N=n, Nmiss =nmiss, Nmiss_pct = nmiss_pct, sum=sum, avg=mean, meidan=median, std=std, var=var, range=range, pctl=pctl))
}

#outlier_treatment
outlier_treatment <- function(x){
  UC = quantile(x, p=0.99, na.rm=T)
  LC = quantile(x, p=0.01, na.rm=T)
  x = ifelse(x>UC, UC, x)
  x = ifelse(x<LC, LC, x)
  return(x)
}

#missing_value_treatment continuous
missing_value_treatment = function(x){
  x[is.na(x)] = mean(x, na.rm=T)
  return(x)
}

#mode for categorical
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#missing_value_treatment categorical
missing_value_treatment_categorical <- function(x){
  x[is.na(x)] <- Mode(na.omit(x))
  return(x)
}

#*#**************************************************************************************************************#

#######################
#-->Data Treatment <--#
#######################

cont_col <- c(colnames(select_if(car_sales,is.numeric)))
cat_col <- colnames(select_if(car_sales,is.character))

car_sales_cont <- car_sales[,cont_col]
car_sales_cat <- car_sales[,cat_col]

names(car_sales_cat)

sum(car_sales_cat$Model == "n/a")
sum(car_sales_cat$Model == "na")
sum(car_sales_cat$Model == "nill")
sum(car_sales_cat$Model == "NULL")
sum(car_sales_cat$Model == "!VALUE#")
sum(car_sales_cat$Model == "")

#Outlier Treatment & Missing Value treatment for continuous variables

num_sum <- data.frame(t(round(apply(car_sales_cont,2,cont_var_summary),2)))

car_sales_cont <- data.frame(apply(car_sales_cont,2,outlier_treatment))
car_sales_cont <- data.frame(apply(car_sales_cont,2,missing_value_treatment))

#Mode Treatment for categorical variables

car_sales_cat <- data.frame(apply(car_sales_cat,2,missing_value_treatment_categorical))

#*#**************************************************************************************************************#

##########################
#--> Dummies Creation <--#
##########################

car_sales_cat <- fastDummies::dummy_cols(car_sales_cat,remove_first_dummy = TRUE)

car_sales_cat <- select(car_sales_cat,-cat_col)

car_sales_clean <- cbind(car_sales_cont,car_sales_cat)

#*#**************************************************************************************************************#

#####################
#--> ASSUMPTIONS <--#
#####################

#target should be ND

ggplot(car_sales_clean) + aes(Sales_in_thousands) + geom_histogram(bins = 10,fill = 'blue',color = 'white')

#To normalise we take Log

car_sales_clean['ln_Sales_in_thousands'] <- log(car_sales_clean$Sales_in_thousands)

ggplot(car_sales_clean) + aes(ln_Sales_in_thousands) + geom_histogram(bins = 10,fill = 'blue',color = 'white')

#Corelation Between x & y ,x & x variables

corel_matrix <- data.frame(round(cor(car_sales_clean),2))

#four_year_resale_value has high corelation so to be dropped

#*#**************************************************************************************************************#


###########################
#--> Feature Reduction <--#
###########################

feat <- data.matrix(select(car_sales_clean,-Sales_in_thousands))
target <- data.matrix(select(car_sales_clean,Sales_in_thousands))

set.seed(12345)

#--> Stepwise <--#

#Full & Empty model
m_full <- lm(Sales_in_thousands~.,data = car_sales_clean)
m_null <- lm(Sales_in_thousands~1,data = car_sales_clean)

stepwise_feat <- step(m_null,scope = list(upper = m_full),data = car_sales_clean, direction = 'both')

# ln_Sales_in_thousands + Manufacturer_Ford + 
# Manufacturer_Honda + Manufacturer_Dodge + Price_in_thousands + 
# Curb_weight + Manufacturer_Toyota + Wheelbase + Manufacturer_Jeep

#--> RFE <--#

rfe_model <- caret::rfe(feat, target, size = c(1:42), rfeControl=rfeControl(functions = lmFuncs))
#size - No. of columns in features
#Warning in RFE is normal

rfe_feat <- update(rfe_model,feat,target,size = 10)
rfe_feat[["bestVar"]]

#RFE not working since no. of observations is less 

feat_selected <- c('ln_Sales_in_thousands','Manufacturer_Ford',
                   'Manufacturer_Honda','Manufacturer_Dodge','Price_in_thousands',
                   'Curb_weight','Manufacturer_Toyota','Wheelbase','Manufacturer_Jeep','Sales_in_thousands')

feat_selected <- feat_selected[!duplicated(feat_selected)]

car_sales_clean_selected <- car_sales_clean[,feat_selected]

#--> LASSO <--#
lasso = train(Sales_in_thousands~.,
              data=car_sales_clean_selected,method='glmnet',
              trControl = trainControl(method="none"),
              tuneGrid=expand.grid(alpha=1,lambda=0.01))

coef(lasso$finalModel, s = lasso$bestTune$lambda)

#--> VIF <--#
m_full <- lm(Sales_in_thousands~.,data = car_sales_clean_selected)
vif(m_full)

#*#**************************************************************************************************************#

########################
#--> Data Splitting <--#
########################

samp <- sample(1:nrow(car_sales_clean_selected), floor(nrow(car_sales_clean_selected)*0.7))

dev <-car_sales_clean_selected[samp,]
val <-car_sales_clean_selected[-samp,]

#*#**************************************************************************************************************#

########################
#--> Model Building <--#
########################

M0 <- lm(Sales_in_thousands~ln_Sales_in_thousands+
           Manufacturer_Ford+
           Price_in_thousands+
           Curb_weight,data = dev)

summary(M0)

#--> Columns Removed <--#
#Wheelbase
#Manufacturer_Jeep
#Manufacturer_Toyota
#Manufacturer_Honda
#Manufacturer_Dodge

dev <- data.frame(cbind(dev,pred = predict(M0)))
val <- data.frame(cbind(val,pred = predict(M0, newdata = val)))

#*#**************************************************************************************************************#

#*#####################
#--> Model Scoring <--#
#######################

#--> MAPE <--#
mape(dev$Sales_in_thousands,dev$pred)
mape(val$Sales_in_thousands,val$pred)

#--> RMSE <--#
rmse(dev$Sales_in_thousands,dev$pred)
rmse(val$Sales_in_thousands,val$pred)

#--> R^2 <--#
MLmetrics::R2_Score(dev$pred,dev$Sales_in_thousands)
MLmetrics::R2_Score(val$pred,val$Sales_in_thousands)

#*#*#**************************************************************************************************************#

#*#######################
#--> Cook's Distance <--#
#########################

#To Reduce Error

cd <- cooks.distance(M0)

plot(cd,pch = '*',cex = 2,main = 'Influencers')
abline(h = 4/nrow(dev),col = 'red')

#Remove Influential outliers
influerncers <- as.numeric(names(cd)[cd>(4/nrow(dev))])

dev2 <- dev[-influerncers,]
dev2$pred <- NULL
val$pred <- NULL

M1 <- lm(Sales_in_thousands~ln_Sales_in_thousands+
           Manufacturer_Ford+
           Manufacturer_Honda+
           Price_in_thousands+
           Curb_weight,data = dev2)

summary(M1)

dev2 <- data.frame(cbind(dev2,pred = predict(M1)))
val2 <- data.frame(cbind(val,pred = predict(M1, newdata = val)))

#*#*#**************************************************************************************************************#

#*#####################
#--> Model Scoring <--#
#######################

#--> MAPE <--#
mape(dev2$Sales_in_thousands,dev2$pred)
mape(val2$Sales_in_thousands,val2$pred)

#--> RMSE <--#
rmse(dev2$Sales_in_thousands,dev2$pred)
rmse(val2$Sales_in_thousands,val2$pred)

#--> R^2 <--#
MLmetrics::R2_Score(dev2$pred,dev2$Sales_in_thousands)
MLmetrics::R2_Score(val2$pred,val2$Sales_in_thousands)

#*#*#**************************************************************************************************************#

#*#######################
#--> Decile Analysis <--#
#########################

dev2.1 <- dev2

#Deciles
dec <- quantile(dev2.1$pred,probs = seq(0.1,0.9,by=0.1))

#intervals
dev2.1$decile <- findInterval(dev2.1$pred,c(-Inf,dec,Inf))

#to check deciles
xtabs(~decile,dev2.1)

dev2_1 <- dev2.1[,c("decile","Sales_in_thousands","pred")]
colnames(dev2_1) <- c('decile','Sales_in_thousands','pred')

dev_dec <- sqldf::sqldf(
                       " select decile,
                         count(decile) cnt,
                         avg(pred) as avg_pred_Y,
                         avg(Sales_in_thousands) avg
                         from dev2_1
                         group by decile 
                         order by decile")

writexl::write_xlsx(dev_dec,'DA_dev.xlsx')

val2.1 <- val2

#Deciles
dec <- quantile(val2.1$pred,probs = seq(0.1,0.9,by=0.1))

#intervals
val2.1$decile <- findInterval(val2.1$pred,c(-Inf,dec,Inf))

#to check deciles
xtabs(~decile,val2.1)

val2_1 <- val2.1[,c("decile","Sales_in_thousands","pred")]
colnames(val2_1) <- c('decile','Sales_in_thousands','pred')

val_dec <- sqldf::sqldf(
                       " select decile,
                         count(decile) cnt,
                         avg(pred) as avg_pred_Y,
                         avg(Sales_in_thousands) avg
                         from val2_1
                         group by decile 
                         order by decile")                      

writexl::write_xlsx(val_dec,'DA_val.xlsx')

#*#*#**************************************************************************************************************#

#*#########################
#--> Model Diagnostics <--#
###########################

coefficients(M1) # model coefficients
confint(M1, level=0.95) # CIs for model parameters 
fitted(M1) # predicted values
residuals(M1) # residuals
anova(M1) # anova table 
vcov(M1) # covariance matrix for model parameters 
influence(M1) # regression diagnostics

#*#*#**************************************************************************************************************#