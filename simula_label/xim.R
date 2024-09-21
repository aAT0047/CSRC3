data <- simla3r
order_cc  <-as.matrix.data.frame(data[2])
random_cc <- as.matrix.data.frame(data[3])    
adaboostc2<- as.matrix.data.frame(data[4])
library(plotrix)
library(showtext)
font_add("SyHei", "SourceHanSansSC-Bold.otf")
rownames(rbind(t(adaboostc2-random_cc),t(order_cc-random_cc)) = c("adaboostc2", "opt_orderc2")
barplot(rbind(t(adaboostc2-random_cc),t(order_cc-random_cc))[,1:11], main="评估指标"
        ,names.arg=c("acc","precision","recall","f1","hitrate","ubsetAcc","hamming","neerror","overage","rankingloss","avg_precision")
        ,beside = FALSE,legend=TRUE,col=c("blue","green"),  family='SyHei')


