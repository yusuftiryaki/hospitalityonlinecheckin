library(lubridate)
library(tree)
library(ISLR)
library(caret)
library(dplyr)
library(mice)
library(pROC)
library(Amelia)
library(Metrics)

options(echo = FALSE)
options(OutDec = ".", digits = 4)
set.seed(101) 

checkindata <- read.csv("CheckinAdvanceJune30.csv", sep = ";", stringsAsFactors = TRUE)

checkindata$IsLogin <- as.factor(checkindata$IsLogin)
checkindata$EmailRead <- as.factor(checkindata$EmailRead)
checkindata$RoomSelected <- as.factor(checkindata$RoomSelected)
#checkindata$GuestId <- as.factor(checkindata$GuestId)
checkindata$Title <- as.character(checkindata$Title)
checkindata$EmailDomain <- as.character(checkindata$EmailDomain)
checkindata$CountryCode <- as.character(checkindata$CountryCode)



checkindata$BookingDateAsDate = as.Date(checkindata$BookingDate,"%d.%m.%Y")
checkindata[grep("-", checkindata$BookingDate),"BookingDateAsDate"] = as.Date( checkindata[grep("-", checkindata$BookingDate),"BookingDate"])

checkindata$ArrivalDateAsDate = as.Date(checkindata$ArrivalDate,"%d.%m.%Y")
checkindata[grep("-", checkindata$ArrivalDate),"ArrivalDateAsDate"] = as.Date( checkindata[grep("-", checkindata$ArrivalDate),"ArrivalDate"])

#new features
checkindata$BookingWindow =   scale(checkindata$ArrivalDateAsDate - checkindata$BookingDateAsDate)[,1]
checkindata$BookingDayOfYear = yday(checkindata$BookingDateAsDate)
checkindata$ArrivalDayOfYear = yday(checkindata$ArrivalDateAsDate)

checkindata$Rate =  as.numeric(sub(",",".",checkindata$Rate))
checkindata$RatePerNight = checkindata$Rate/ifelse(checkindata$Nights == 0,-1,checkindata$Nights)

#remove unmeaningfull features
checkindata = checkindata[, !names(checkindata) %in% c("IsLogin","Title","BookingDate", "ArrivalDate","BookingDateAsDate","ArrivalDateAsDate")] 

#impute missing values with mice pack
missmap(checkindata[,-1],
        main = "Missing values in  Dataset",
        y.labels = NULL,
        y.at = NULL)
checkindata.imp <- mice(checkindata, m=1, method='cart', printFlag=FALSE)
checkindata = complete(checkindata.imp)







#impute missing values with rf
#checkindata.learn <- rfImpute(RoomSelected ~ ., checkindata.learn)

#sort(sapply(checkindata, function(x) { sum(is.na(x)) }), decreasing=TRUE)



