# performed linear mixed-effect model on features
library('lme4')
library('lmerTest')

data <- read.csv('[data directory]')

head(data)

data_1 <- subset(data, Agglomerative==1)
data_2 <- subset(data, Agglomerative==0)
data_3 <- subset(data, Agglomerative==2)

# updrs1
res_1 <- lmer(updrs1 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(updrs1 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(updrs1 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# hallucination
res_1 <- lmer(hallucination ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(hallucination ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(hallucination ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# Apathy
res_1 <- lmer(Apathy ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(Apathy ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(Apathy ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# Pain
res_1 <- lmer(Pain ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(Pain ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(Pain ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# Fatigue
res_1 <- lmer(Fatigue ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(Fatigue ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(Fatigue ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# epworth
res_1 <- lmer(epworth ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(epworth ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(epworth ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# GDS
res_1 <- lmer(GDS ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(GDS ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(GDS ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# stai_state
res_1 <- lmer(stai_state ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(stai_state ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(stai_state ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# stai_trait
res_1 <- lmer(stai_trait ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(stai_trait ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(stai_trait ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# moca
res_1 <- lmer(moca ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(moca ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(moca ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# benton
res_1 <- lmer(benton ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(benton ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(benton ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# HVLT_total_recall
res_1 <- lmer(HVLT_total_recall ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(HVLT_total_recall ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(HVLT_total_recall ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# HVLT_Discrimination_Recognition
res_1 <- lmer(HVLT_Discrimination_Recognition ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(HVLT_Discrimination_Recognition ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(HVLT_Discrimination_Recognition ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# HVLT_Retention
res_1 <- lmer(HVLT_Retention ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(HVLT_Retention ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(HVLT_Retention ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# HVLT_delayed_recall
res_1 <- lmer(HVLT_Delayed_Recall ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(HVLT_Delayed_Recall ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(HVLT_Delayed_Recall ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# RBD
res_1 <- lmer(RBD ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(RBD ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(RBD ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# QUIP
res_1 <- lmer(QUIP ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(QUIP ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(QUIP ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# LNS
res_1 <- lmer(LNS ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(LNS ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(LNS ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# Semantic_Fluency
res_1 <- lmer(Semantic_Fluency ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(Semantic_Fluency ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(Semantic_Fluency ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# SDM
res_1 <- lmer(SDM ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(SDM ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(SDM ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# aut
data_1$aut_total <- rowSums(data_1[, c("aut_gastrointestinal_up",	"aut_gastrointestinal_down",	
                                       "aut_urinary",	"aut_cardiovascular",	"aut_thermoregulatory",
                                       "aut_pupillomotor",	"aut_skin",	"aut_sexual")])
data_2$aut_total <- rowSums(data_2[, c("aut_gastrointestinal_up",	"aut_gastrointestinal_down",	
                                       "aut_urinary",	"aut_cardiovascular",	"aut_thermoregulatory",
                                       "aut_pupillomotor",	"aut_skin",	"aut_sexual")])
data_3$aut_total <- rowSums(data_3[, c("aut_gastrointestinal_up",	"aut_gastrointestinal_down",	
                                       "aut_urinary",	"aut_cardiovascular",	"aut_thermoregulatory",
                                       "aut_pupillomotor",	"aut_skin",	"aut_sexual")])

data_1$aut_gastrointestinal <- rowSums(data_1[, c("aut_gastrointestinal_up",	"aut_gastrointestinal_down")])
data_2$aut_gastrointestinal <- rowSums(data_2[, c("aut_gastrointestinal_up",	"aut_gastrointestinal_down")])
data_3$aut_gastrointestinal <- rowSums(data_3[, c("aut_gastrointestinal_up",	"aut_gastrointestinal_down")])

res_1 <- lmer(aut_gastrointestinal ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_gastrointestinal ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_gastrointestinal ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


res_1 <- lmer(aut_urinary ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_urinary ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_urinary ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


res_1 <- lmer(aut_cardiovascular ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_cardiovascular ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_cardiovascular ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


res_1 <- lmer(aut_thermoregulatory ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_thermoregulatory ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_thermoregulatory ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



res_1 <- lmer(aut_pupillomotor ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_pupillomotor ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_pupillomotor ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



res_1 <- lmer(aut_skin ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_skin ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_skin ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


res_1 <- lmer(aut_sexual ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_sexual ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_sexual ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



res_1 <- lmer(aut_total ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(aut_total ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(aut_total ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")




# updrs2
res_1 <- lmer(updrs2 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(updrs2 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(updrs2 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# updrs3
res_1 <- lmer(updrs3 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(updrs3 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(updrs3 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# HY_stage
res_1 <- lmer(HY_stage ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(HY_stage ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(HY_stage ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# Schwab
res_1 <- lmer(Schwab ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(Schwab ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(Schwab ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# Tremor_score
res_1 <- lmer(Tremor_score ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(Tremor_score ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(Tremor_score ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# PIGD_score
res_1 <- lmer(PIGD_score ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(PIGD_score ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(PIGD_score ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# alpha_syn
res_1 <- lmer(alpha_syn ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(alpha_syn ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(alpha_syn ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# abeta_42
res_1 <- lmer(abeta_42 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(abeta_42 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(abeta_42 ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# p_tau181p
res_1 <- lmer(p_tau181p ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(p_tau181p ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(p_tau181p ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# total_tau
res_1 <- lmer(total_tau ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(total_tau ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(total_tau ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")



# abeta_42_total_tau_ratio
res_1 <- lmer(abeta_42_total_tau_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(abeta_42_total_tau_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(abeta_42_total_tau_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# abeta_42_alpha_syn_ratio
res_1 <- lmer(abeta_42_alpha_syn_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(abeta_42_alpha_syn_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(abeta_42_alpha_syn_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")


# p_tau181p_alpha_syn_ratio
res_1 <- lmer(p_tau181p_alpha_syn_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_1)
summary(res_1)
confint(res_1, method="Wald")
#confint.merMod(res_1, method = "boot", boot.type = "perc", nsim = 500, oldNames = F)

res_2 <- lmer(p_tau181p_alpha_syn_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_2)
summary(res_2)
confint(res_2, method="Wald")

res_3 <- lmer(p_tau181p_alpha_syn_ratio ~ time + Age_at_symptom + Gender_male + LEDD_clipped_scaled + (1 + time | PATNO), data=data_3)
summary(res_3)
confint(res_3, method="Wald")

