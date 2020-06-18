set.seed(123)

n <- 2000

persons <- c("Ana", "Bruno", "Carla", "Diogo", "Eva", "Francisco")

cats  <- c("curso", "casa", "carro", "comida", "misc")

df <- data.frame(pessoa    = sample(persons, n, replace=TRUE, prob=c(.15,.25,.1,.1,.1,.3)),
                 valor     = round(100*rlnorm(n,0,1)-50,2),
                 data      = (Sys.Date()-365) + sort(sample(1:365, n, rep=TRUE)),
                 categoria = sample(cats, n, replace=TRUE, prob=c(.2,.3,.1,.3,.1))
                )

write.csv(df, file="movimentos.csv", row.names=FALSE)
