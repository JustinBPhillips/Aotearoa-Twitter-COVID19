rm(list=ls())

library(data.table)
library(lubridate)
library(scales)
library(hrbrthemes)
library(ggplot2)
library(dplyr)
library(fs)

setwd(paste(fs::path_home(),'/OneDrive/manuscripts/nzTwitterCovid/data/bert/combine',sep=""))


#### OK, let's get the combo results
library(data.table)
library(stringr)
results_bert = fread("~/nzTwitterCovid/data/bert/results_merged.tsv",header=TRUE,sep="\t",quote=FALSE)

results_all_sentencesAuckland = fread("~/nzTwitterCovid/data/bert/all_sentences_nzTwitterAuckland.tsv",header=TRUE,sep="\t",quote=FALSE)
unique_sentencesAuckland = fread("~/nzTwitterCovid/data/bert/uniques_nzTwitterAuckland.tsv",header=FALSE,sep="\t",quote=FALSE)

results_all_sentencesOutsideAuckland = fread("~/nzTwitterCovid/data/bert/all_sentences_nzTwitterOutsideAuckland.tsv",header=TRUE,sep="\t",quote=FALSE)
unique_sentencesOutsideAuckland = fread("~/nzTwitterCovid/data/bert/uniques_nzTwitterOutsideAuckland.tsv",header=FALSE,sep="\t",quote=FALSE)

auckland_posts = fread("~/nzTwitterCovid/data/bert/tweetsOnlyAuckland.tsv",header=FALSE,sep="\t",quote=FALSE)
auckland_posts$V1 = as.POSIXct(auckland_posts$V1)

outsideAuckland_posts = fread("~/nzTwitterCovid/data/bert/tweetsOnlyOutsideAuckland.tsv",header=FALSE,sep="\t",quote=FALSE)
outsideAuckland_posts$V1 = as.POSIXct(outsideAuckland_posts$V1)

hdbscan_results = fread("~/nzTwitterCovid/data/bert/hdbscan_merged.tsv",header=TRUE,sep="\t",quote=FALSE)
model_aucklandHDBSCAN = fread("~/nzTwitterCovid/data/bert/hdbscan_nzTwitterAuckland_0.002_mcp_200_ms.tsv",header=TRUE,sep="\t",quote=FALSE)
model_aucklandHDBSCAN$clusters = paste("AK_",model_aucklandHDBSCAN$clusters,sep="")
model_OutsideAucklandHDBSCAN = fread("~/nzTwitterCovid/data/bert/hdbscan_nzTwitterOutsideAuckland_0.002_mcp_200_ms.tsv",header=TRUE,sep="\t",quote=FALSE)
model_OutsideAucklandHDBSCAN$clusters = paste("OAK_",model_OutsideAucklandHDBSCAN$clusters,sep="")


hdbscan_resultsAuckland = hdbscan_results[1:length(unique_sentencesAuckland$V1)]
hdbscan_resultsOutsideAuckland = hdbscan_results[(length(unique_sentencesAuckland$V1)+1):length(hdbscan_results$clusters)]

results_all_sentencesAuckland$Date = as.Date(auckland_posts[results_all_sentencesAuckland$index+1]$V1)
results_all_sentencesAuckland$Topic = hdbscan_resultsAuckland[match(results_all_sentencesAuckland$sentence,unique_sentencesAuckland$V1)]
results_all_sentencesAuckland$Model = model_aucklandHDBSCAN[match(results_all_sentencesAuckland$sentence,unique_sentencesAuckland$V1)]
results_all_sentencesAuckland$Location = rep("Auckland",length(results_all_sentencesAuckland$Topic))

results_all_sentencesOutsideAuckland$Date = as.Date(outsideAuckland_posts[results_all_sentencesOutsideAuckland$index+1]$V1)
results_all_sentencesOutsideAuckland$Topic = hdbscan_resultsOutsideAuckland[match(results_all_sentencesOutsideAuckland$sentence,unique_sentencesOutsideAuckland$V1)]
results_all_sentencesOutsideAuckland$Model = model_OutsideAucklandHDBSCAN[match(results_all_sentencesOutsideAuckland$sentence,unique_sentencesOutsideAuckland$V1)]
results_all_sentencesOutsideAuckland$Location = rep("Outside Auckland",length(results_all_sentencesOutsideAuckland$Topic))

results_all_sentences = rbind(results_all_sentencesAuckland,results_all_sentencesOutsideAuckland)


proportions_table = data.table(hdbscan=c(),models=c(),size=c())
english_classified = filter(results_all_sentences,Topic!="-1")

cases = fread("~/OneDrive/manuscripts/nzTwitterCovid/data/WHO-COVID-19-global-data.csv")
nzCovidTime = filter(cases,Country=='New Zealand',Date_reported>=as.Date("2020-03-15", tz = "NZ"),Date_reported<=as.Date("2022-04-14", tz = "NZ"))


### draw up initial timeline labels	

  

labellingFunction <- function(tmp,ymin,ymax) {
  
  tmp = tmp + geom_segment(x = as.numeric(as.Date("2020-03-25", tz = "NZ")),xend = as.numeric(as.Date("2020-03-25", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    geom_segment(x = as.numeric(as.Date("2020-05-11", tz = "NZ")),xend = as.numeric(as.Date("2020-05-11", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2020-03-25", tz = "NZ"), xmax=as.Date("2020-05-11", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1,fill="blue")
    
  
  tmp = tmp + geom_segment(x = as.numeric(as.Date("2020-08-14", tz = "NZ")),xend = as.numeric(as.Date("2020-08-14", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    geom_segment(x = as.numeric(as.Date("2020-09-23", tz = "NZ")),xend = as.numeric(as.Date("2020-09-23", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2020-08-14", tz = "NZ"), xmax=as.Date("2020-09-23", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1)+
    geom_segment(x = as.numeric(as.Date("2020-10-17", tz = "NZ")),xend = as.numeric(as.Date("2020-10-17", tz = "NZ")), y=ymin, yend=ymax,color="blue",linetype=3,colour="blue",alpha=0.5)
    
  
  tmp = tmp + geom_segment(x = as.numeric(as.Date("2021-02-14", tz = "NZ")),xend = as.numeric(as.Date("2021-02-14", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    geom_segment(x = as.numeric(as.Date("2021-02-22", tz = "NZ")),xend = as.numeric(as.Date("2021-02-22", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2021-02-14", tz = "NZ"), xmax=as.Date("2021-02-22", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1)
  
  tmp = tmp + geom_segment(x = as.numeric(as.Date("2021-02-14", tz = "NZ")),xend = as.numeric(as.Date("2021-02-14", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    geom_segment(x = as.numeric(as.Date("2021-02-22", tz = "NZ")),xend = as.numeric(as.Date("2021-02-22", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2021-02-14", tz = "NZ"), xmax=as.Date("2021-02-22", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1)
  
  tmp = tmp + geom_segment(x = as.numeric(as.Date("2021-02-28", tz = "NZ")),xend = as.numeric(as.Date("2021-02-28", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    geom_segment(x = as.numeric(as.Date("2021-03-12", tz = "NZ")),xend = as.numeric(as.Date("2021-03-12", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2021-02-28", tz = "NZ"), xmax=as.Date("2021-03-12", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1)
  
  tmp = tmp + geom_segment(x = as.numeric(as.Date("2021-08-17", tz = "NZ")),xend = as.numeric(as.Date("2021-08-17", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+ 
    geom_segment(x = as.numeric(as.Date("2021-09-07", tz = "NZ")),xend = as.numeric(as.Date("2021-09-07", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    geom_segment(x = as.numeric(as.Date("2021-12-02", tz = "NZ")),xend = as.numeric(as.Date("2021-12-02", tz = "NZ")), y=ymin, yend=ymax, linetype=4,color="black",alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2021-08-17", tz = "NZ"), xmax=as.Date("2021-12-02", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1,alpha=0.5)+
    ggplot2::annotate("rect",xmin=as.Date("2021-08-17", tz = "NZ"), xmax=as.Date("2021-09-07", tz = "NZ"), ymin=ymin, ymax=ymax,alpha=0.1,fill="blue",alpha=0.5)
  
  return(tmp)
}


### draw up proportions table and create thumbnail graphs
library(lubridate)
library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(scales)
library(data.table)
library(Rbeast)
library(lubridate)
library(tidyr)
library(dplyr)
customPalette = c("#176ae2","#007113")
my_theme <- function(){
  list(
    theme_ipsum_rc(),
    scale_color_manual(values = customPalette),
    scale_fill_manual(values = customPalette),
    scale_linetype_manual(values = c(1,1))
  )
}

# I need to do reverse for legend png below
topics = rev(unique(english_classified$Topic))
for(cluster in topics) {
  full_cluster = filter(results_all_sentences,Topic==cluster)
  cluster_sentences = filter(full_cluster,Topic==cluster,Date>=as.Date("2020-03-15", tz = "NZ"),Date<=as.Date("2022-04-14", tz = "NZ"))
  auckCluster = filter(cluster_sentences,Location=='Auckland')
  outsideAuckCluster = filter(cluster_sentences,Location=='Outside Auckland')
  
  theGraphData <- cluster_sentences %>% group_by(Date,Location) %>% summarize(Posts = n())
  # normalize
  auckCluster = filter(theGraphData,Location=='Auckland')
  auckCluster$Posts = auckCluster$Posts/max(auckCluster$Posts)
  outsideAuckCluster = filter(theGraphData,Location=='Outside Auckland')
  outsideAuckCluster$Posts = outsideAuckCluster$Posts/max(outsideAuckCluster$Posts)
  theGraphData = rbind(auckCluster,outsideAuckCluster)

  tmp = ggplot(data=theGraphData,aes(x=Date,y=Posts,color=Location)) +
    geom_col(data=nzCovidTime,aes(x=Date_reported,y=New_cases/24000), size=.1, color="darkgrey", alpha=.1)+
    stat_smooth(method="loess",size=1.2,se=FALSE,span=0.1,aes(linetype=Location))+
    my_theme()+theme(legend.position = "none")+
    scale_x_date(breaks = "1 year", date_labels =  "%Y",name="")+
    scale_y_continuous(name="Posts (normalized)",limits = c(0,1),sec.axis = sec_axis(~.*24000,name="NZ COVID-19 Cases"))+
    theme(legend.position = "none",axis.text.x=element_text(size=12),axis.title.y = element_blank(),axis.text.y = element_blank(),axis.title.x=element_blank(),plot.margin = margin(t = 0.1,r = 0.1,b = 0.1,l = 0.1,unit = "mm"))
  tmp = labellingFunction(tmp,min(theGraphData$Posts),max(theGraphData$Posts))

  
  for(location in unique(theGraphData$Location)) {
    original_df = theGraphData %>% filter(Location==location) %>% select(-Location) 
    original_df$Date = as.Date(original_df$Date)
    df = original_df %>% group_by(Date) %>% summarize(Posts = Posts)
    df = df %>% complete(Date = seq(min(Date), max(Date),by = "1 day"))
    df = df[order(df$Date),]
    out=beast(df$Posts, season='none',mcmc.seed=42,hasOutlier = TRUE)
    ragg::agg_png(paste("./thumbnails/beast_diagnostic_",location,cluster,".png",sep=""), width = 1050, height = 700, units = "px", res = 300, scaling=0.35)
    plot(out)
    dev.off()
    
    sink(file=paste("./thumbnails/beast_diagnostic",location,cluster,".txt",sep=""))
    print(out)
    
    for(i in 1:length(unlist(out$trend["cpPr"]))) {
      cp = unlist(out$trend["cpPr"])[i]
      if(!is.na(cp) & cp>0.99) {
        point = original_df[which(original_df$Date==df[unlist(out$trend["cp"])[i],]$Date),]
        print(paste(point$Date,point$Posts))
        tmp = tmp + geom_point(size=5,shape=13,data=data.table(Date=c(point$Date),Posts=c(point$Posts),Location=c(location)))
      }
    }
    
    
    for(i in 1:length(unlist(out$outlier["cpPr"]))) {
      cp = unlist(out$outlier["cpPr"])[i]
      if(!is.na(cp) & cp>0.99) {
        point = original_df[which(original_df$Date==df[unlist(out$outlier["cp"])[i],]$Date),]
        print(paste(point$Date,point$Posts))
        tmp = tmp + geom_point(size=5,shape=7,data=data.table(Date=c(point$Date),Posts=c(point$Posts),Location=c(location)))
      }
    }
    
    sink(file=NULL)
    
  }
  
  
  
  ragg::agg_png(paste("./thumbnails/timeseries_",cluster,".png",sep=""), width = 176, height = 99, units = "px", res = 60, scaling=1)
  print(tmp)
  dev.off()
  
  tmp = tmp + theme(legend.position = "hide",legend.text = element_text(size=12), axis.text.x= element_text(size=13),axis.title.y = element_text(size=13,angle=90),axis.text.y = element_text(size=13),axis.title.x=element_blank(),plot.margin = margin(t = 2,r = 0.1,b = 0.1,l = 0.1,unit = "mm"))
  ragg::agg_png(paste("./thumbnails/timeseries_medium_",cluster,".png",sep=""), width = 352, height = 198, units = "px", res = 60, scaling=1)
  print(tmp)
  dev.off()
  
  tmp = tmp + theme(legend.position = "top",legend.text = element_text(size=12), axis.text.x= element_text(size=13),axis.title.y = element_text(size=13,angle=90),axis.text.y = element_text(size=13),axis.title.x=element_blank(),plot.margin = margin(t = 0.1,r = 0.1,b = 0.1,l = 0.1,unit = "mm"))
  tmp = tmp + ggplot2::annotate(geom="text",x=as.Date("2020-03-25", tz = "NZ")-10,y=max(theGraphData$Posts)*0.85,label="NZ-wide lockdown",angle=90)
  tmp = tmp + ggplot2::annotate(geom="text",x=as.Date("2020-08-14", tz = "NZ")-10,y=max(theGraphData$Posts)*0.85,label="Auckland lockdown",angle=90)
  tmp = tmp + ggplot2::annotate(geom="text",x=as.Date("2020-10-17", tz = "NZ")-10,y=max(theGraphData$Posts)*0.85,label="General election",angle=90)
  
  ragg::agg_png(paste("./thumbnails/timeseries_large_",cluster,".png",sep=""), width = 704, height = 396, units = "px", res = 70, scaling=1)
  print(tmp)
  dev.off()
  
  
  topic_size = length(full_cluster$Topic)/length(results_all_sentences$Topic)
  
  proportions_table = rbind(proportions_table,data.table(hdbscan=c(cluster),models=c(paste(unique(cluster_sentences$Model),collapse=" ")),size=c(topic_size)))
}


library(grid)
library(gridExtra) 
tmp_timeseriesLEGEND = tmp + theme(legend.position = "right",legend.title=element_blank(),legend.text = element_text(family='serif',size=20),legend.key.spacing.x = unit(7, "mm"),legend.margin=margin(0, 18, 0, 0))
legend <- cowplot::get_plot_component(tmp_timeseriesLEGEND, 'guide-box-right', return_all = TRUE)
tmplabel = grid.newpage()
ragg::agg_png(paste("./thumbnails/timeseries_legend.png"), width = 176, height = 30, units = "px", res = 50, scaling=1)
grid.draw(legend)
dev.off()




# export results and images to html
library(xtable)

results_bertAuckland = fread("~/nzTwitterCovid/data/bert/results_hdbscan_nzTwitterAuckland_0.002_mcp_200_ms.tsv",header=TRUE,sep="\t",quote=FALSE)
results_bertOutsideAuckland = fread("~/nzTwitterCovid/data/bert/results_hdbscan_nzTwitterOutsideAuckland_0.002_mcp_200_ms.tsv",header=TRUE,sep="\t",quote=FALSE)

tmp_results_bert = results_bert
tmp_results_bert$Timeline = paste("<img src=\"",getwd(),"/thumbnails/timeseries_",results_bert$Topic,".png\" /img>",sep="")
# remove noise category
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==-1)),]
combostuff = data.table(oldmodels=c(),repdocs=c())
for (topic in tmp_results_bert$Topic) {
  old_models = proportions_table[which(proportions_table$hdbscan==topic)]$models
  repdocs = c()
  for (model in str_split_1(old_models," ")) {
    if(startsWith(model,"AK_")) {
      repdocs = c(repdocs,results_bertAuckland[which(results_bertAuckland$Topic==str_split_1(model,"_")[2])]$Representative_Docs)
    } 
    if(startsWith(model,"OAK_")) {
      repdocs = c(repdocs,results_bertOutsideAuckland[which(results_bertOutsideAuckland$Topic==str_split_1(model,"_")[2])]$Representative_Docs)
    } 
  }
  repdocs = paste(repdocs,collapse=" | ")
  combostuff = rbind(combostuff,data.table(oldmodels=c(old_models),repdocs=c(repdocs)))
}
tmp_results_bert$Topic = paste(tmp_results_bert$Topic,combostuff$oldmodels,sep=" ")
tmp_results_bert$Representative_Docs = combostuff$repdocs

tmp_html = gsub(pattern="/img&gt;", replacement="/>",gsub(pattern="&lt;img",replacement="<img",print(xtable(tmp_results_bert), include.rownames=FALSE,type="html")))
# no need for timeseries legend
replacementString = paste("<th> Timeline <br><img src='",getwd(),"/thumbnails/timeseries_legend.png' /img></th>",sep="")
tmp_html = gsub(pattern="<th> Timeline </th>", replacement=replacementString,tmp_html)
cat(tmp_html,file="./htmlTables/combined_models.html")







