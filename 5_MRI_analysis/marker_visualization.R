library(ggplot2)
library(ggseg)
library(ggpubr)
library(extrafont)

thickness_plot_0_1_data = read.csv('[your directory]/thickness_0_1_t.csv', check.names = FALSE)
thickness_plot_0_1_data = thickness_plot_0_1_data[thickness_plot_0_1_data[, 'p-value'] < 0.05, ]
thickness_plot_0_1_data$importance = -log10(thickness_plot_0_1_data$`p-value`)
thickness_plot_0_1_data = thickness_plot_0_1_data[, c('region', 'importance')]
thickness_plot_0_1_data$type = rep('Subtype I vs. Subtype II', dim(thickness_plot_0_1_data)[1])

thickness_plot_0_2_data = read.csv('[your directory]/thickness_0_2_t.csv', check.names = FALSE)
thickness_plot_0_2_data = thickness_plot_0_2_data[thickness_plot_0_2_data[, 'p-value'] < 0.05, ]
thickness_plot_0_2_data$importance = -log10(thickness_plot_0_2_data$`p-value`)
thickness_plot_0_2_data = thickness_plot_0_2_data[, c('region', 'importance')]
thickness_plot_0_2_data$type = rep('Subtype II vs. Subtype III', dim(thickness_plot_0_2_data)[1])

thickness_plot_1_2_data = read.csv('[your directory]/thickness_1_2_t.csv', check.names = FALSE)
thickness_plot_1_2_data = thickness_plot_1_2_data[thickness_plot_1_2_data[, 'p-value'] < 0.05, ]
thickness_plot_1_2_data$importance = -log10(thickness_plot_1_2_data$`p-value`)
thickness_plot_1_2_data = thickness_plot_1_2_data[, c('region', 'importance')]
thickness_plot_1_2_data$type = rep('Subtype I vs. Subtype III', dim(thickness_plot_1_2_data)[1])


thickness_plot_data = rbind(thickness_plot_0_1_data, thickness_plot_0_2_data, thickness_plot_1_2_data)
thickness_plot_data = thickness_plot_data %>% group_by(type)
thickness_plot_data$type = factor(thickness_plot_data$type, levels = c('Subtype I vs. Subtype II', 'Subtype II vs. Subtype III', 'Subtype I vs. Subtype III'))


wmv_plot_0_1_data = read.csv('[your directory]/wmv_0_1_t.csv', check.names = FALSE)
wmv_plot_0_1_data = wmv_plot_0_1_data[wmv_plot_0_1_data[, 'p-value'] < 0.05, ]
wmv_plot_0_1_data$importance = -log10(wmv_plot_0_1_data$`p-value`)
wmv_plot_0_1_data = wmv_plot_0_1_data[, c('region', 'importance')]
wmv_plot_0_1_data$type = rep('Subtype I vs. Subtype II', dim(wmv_plot_0_1_data)[1])

wmv_plot_0_2_data = read.csv('[your directory]/wmv_0_2_t.csv', check.names = FALSE)
wmv_plot_0_2_data = wmv_plot_0_2_data[wmv_plot_0_2_data[, 'p-value'] < 0.05, ]
wmv_plot_0_2_data$importance = -log10(wmv_plot_0_2_data$`p-value`)
wmv_plot_0_2_data = wmv_plot_0_2_data[, c('region', 'importance')]
wmv_plot_0_2_data$type = rep('Subtype II vs. Subtype III', dim(wmv_plot_0_2_data)[1])

wmv_plot_1_2_data = read.csv('[your directory]/wmv_1_2_t.csv', check.names = FALSE)
wmv_plot_1_2_data = wmv_plot_1_2_data[wmv_plot_1_2_data[, 'p-value'] < 0.05, ]
wmv_plot_1_2_data$importance = -log10(wmv_plot_1_2_data$`p-value`)
wmv_plot_1_2_data = wmv_plot_1_2_data[, c('region', 'importance')]
wmv_plot_1_2_data$type = rep('Subtype I vs. Subtype III', dim(wmv_plot_1_2_data)[1])


wmv_plot_data = rbind(wmv_plot_0_1_data, wmv_plot_0_2_data, wmv_plot_1_2_data)
wmv_plot_data = wmv_plot_data %>% group_by(type)
wmv_plot_data$type = factor(wmv_plot_data$type, levels = c('Subtype I vs. Subtype II', 'Subtype II vs. Subtype III', 'Subtype I vs. Subtype III'))

plot_wmv = ggseg(hemisphere='left', colour='black', .data=wmv_plot_data, mapping=aes(fill=importance)) +
  facet_wrap(~type, nrow=1) + 
  scale_fill_gradient2(low="blue", mid='white', high="red", midpoint = 0, na.value = "white", name = "Importance", limits=c(1, 4)) +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, family="Arial"), legend.text=element_text(size=11, family="Arial"),
        axis.text = element_blank(), axis.title = element_blank(),
        strip.text.x = element_text(size = 12, face = "bold", family="Arial"),
        legend.key.height = unit(.3, "cm"), legend.key.width = unit(1, "cm"),
        legend.title = element_text(margin = margin(b = 10), family="Arial", face = "bold"))

plot_thickness = ggseg(hemisphere='left', colour='black', .data=thickness_plot_data, mapping=aes(fill=importance)) +
  facet_wrap(~type, nrow=1) + 
  scale_fill_gradient2(low="blue", mid='white', high="red", midpoint = 0, na.value = "white", name = "Importance", limits=c(1, 4)) +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, family="Arial"), legend.text=element_text(size=11, family="Arial"),
        axis.text = element_blank(), axis.title = element_blank(),
        strip.text.x = element_text(size = 12, face = "bold", family="Arial"),
        legend.key.height = unit(.3, "cm"), legend.key.width = unit(1, "cm"),
        legend.title = element_text(margin = margin(b = 10), family="Arial", face = "bold"))



ggarrange(plot_thickness, plot_wmv, ncol=1, nrow=2,
          labels = c('Thickness', 'White Matter Volume'), hjust = -0.06, label.y = 0.95,
          font.label = list(size = 14, color = "black", face = "bold", family = 'Arial'),
          common.legend = FALSE, legend = 'bottom')

ggsave(
  ggarrange(plot_thickness, plot_wmv, ncol=1, nrow=2,
            labels = c('Thickness', 'White Matter Volume'), hjust = -0.06, label.y = 0.95,
            font.label = list(size = 14, color = "black", face = "bold", family = 'Arial'),
            common.legend = TRUE, legend = 'bottom', legend.grob = get_legend(plot_wmv)),
  filename = '[your directory]/brain_plot.pdf', width = 8, height = 6
)


  
  