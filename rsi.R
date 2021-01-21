daily_data <- calltable('bydaym2', 'y')
# head(daily_data)

stocklist <- unique(daily_data$stock)
# stocklist <- stocklist[1:2]

bb <- data.frame()
for (sl in stocklist) {
  idx <- which(daily_data$stock == sl)
  sdata <- daily_data[idx, ]
  rownames(sdata) <- 1:nrow(sdata)
  
  
  # matrdgraph(daily_data, 30, 60, "2017-04-01")
  
  bb1 <- bestrsi(sdata)
  
  bb <- rbind(bb, bb1)
  bb  
}
