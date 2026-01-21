library(rstan)
library(dplyr)
library(rstudioapi)
library(ggplot2)

# Read Data
train <- read.csv("C:/Users/dengrh/Documents/Work/OSF/Training_50%.csv")
test <- read.csv("C:/Users/dengrh/Documents/Work/OSF/Testing_50%.csv")

train <- read.csv("C:/Users/dengrh/Documents/Work/OSF/Training_80%.csv")
test <- read.csv("C:/Users/dengrh/Documents/Work/OSF/Testing_20%.csv")

spike_slab_stan_code <- "
data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N,P] X;
  int<lower=0, upper=1> y[N];
  real<lower=0> tau0;
  real<lower=0> tau1;
  real<lower=0, upper=1> pi;
}
parameters {
  real alpha;
  vector[P] beta;
  vector<lower=0, upper=1>[P] lambda;
}
model {
  alpha ~ normal(0, 5);
  
  for (j in 1:P) {
    real tau2 = square(tau0) * (1 - lambda[j]) + square(tau1) * lambda[j];
    beta[j] ~ normal(0, sqrt(tau2));
    lambda[j] ~ beta(1, (1 - pi) / pi);  // shrink toward 0
  }
  
  y ~ bernoulli_logit(alpha + X * beta);
}
"

X_train <- select(train, c("VAR_StressDomain_CAT_Red",
                           "VAR_StressDomain_CAT_Blue", 
                           "VAR_SDOHSocialConnectionDomain_CAT_Red", 
                           "VAR_SDOHSocialConnectionDomain_CAT_Orange", 
                           "VAR_SDOHSocialConnectionDomain_CAT_Blue", 
                           "VAR_PhysicalActivityDomain_CAT_Red", 
                           "VAR_PhysicalActivityDomain_CAT_Orange", 
                           "VAR_PhysicalActivityDomain_CAT_Blue", 
                           "VAR_SDOHAlcoholUseDomainRisk_CAT_Red", 
                           "VAR_SDOHAlcoholUseDomainRisk_CAT_Blue", 
                           "VAR_UtilitiesDomain_FLG", 
                           "VAR_TransportationDomainCollected_FLG",
                           "VAR_SafetyandDomesticViolenceDomainCollected_FLG", 
                           "VAR_HousingStabilityDomainCollected_FLG", 
                           "VAR_FoodInsecurityDomainCollected_FLG", 
                           "VAR_FinancialResourceStrainDomainCollected_FLG"))
X_test <- select(test, c("VAR_StressDomain_CAT_Red",
                         "VAR_StressDomain_CAT_Blue", 
                         "VAR_SDOHSocialConnectionDomain_CAT_Red", 
                         "VAR_SDOHSocialConnectionDomain_CAT_Orange", 
                         "VAR_SDOHSocialConnectionDomain_CAT_Blue", 
                         "VAR_PhysicalActivityDomain_CAT_Red", 
                         "VAR_PhysicalActivityDomain_CAT_Orange", 
                         "VAR_PhysicalActivityDomain_CAT_Blue", 
                         "VAR_SDOHAlcoholUseDomainRisk_CAT_Red", 
                         "VAR_SDOHAlcoholUseDomainRisk_CAT_Blue", 
                         "VAR_UtilitiesDomain_FLG", 
                         "VAR_TransportationDomainCollected_FLG",
                         "VAR_SafetyandDomesticViolenceDomainCollected_FLG", 
                         "VAR_HousingStabilityDomainCollected_FLG", 
                         "VAR_FoodInsecurityDomainCollected_FLG", 
                         "VAR_FinancialResourceStrainDomainCollected_FLG"))
y_train <- train[["OUTCOME_BINARY"]]
y_test <- test[["OUTCOME_BINARY"]]
N <- nrow(X_train)
P <- ncol(X_train)

compiled_spike <- stan_model(model_code = spike_slab_stan_code)
stan_data <- list(N = N, P = P, X = X_train, y = y_train, tau0 = 0.01, tau1 = 5, pi = 0.2)
options(mc.cores = parallel::detectCores())

fit <- sampling(compiled_spike, data = stan_data, iter = 2000, chains = 4)

# Accuracy and confusion matrix
post <- extract(fit)
beta_hat <- colMeans(post$beta)
alpha_hat <- mean(post$alpha)

X_mat <- as.matrix(X_test)

logit <- function(x) 1 / (1 + exp(-x))
prob_pred <- logit(alpha_hat + X_mat %*% matrix(beta_hat, ncol = 1))
y_pred <- ifelse(prob_pred > 0.5, 1, 0)
accuracy <- mean(y_pred == y_test)
print(paste("Test accuracy:", round(accuracy, 4)))
table(Predicted = y_pred, Actual = y_test)

evaluate_stan_logistic <- function(fit, X_test, y_test, threshold = 0.5, show_auc = TRUE) {
  post <- rstan::extract(fit)
  beta_hat <- colMeans(post$beta)
  alpha_hat <- mean(post$alpha)
  
  X_mat <- as.matrix(X_test)
  beta_vec <- matrix(beta_hat, ncol = 1)
  
  logit <- function(x) 1 / (1 + exp(-x))
  
  prob_pred <- logit(alpha_hat + X_mat %*% beta_vec)
  y_pred <- ifelse(prob_pred > threshold, 1, 0)
  
  conf <- table(Predicted = y_pred, Actual = y_test)
  TP <- ifelse("1" %in% rownames(conf) & "1" %in% colnames(conf), conf["1", "1"], 0)
  FP <- ifelse("1" %in% rownames(conf) & "0" %in% colnames(conf), conf["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(conf) & "1" %in% colnames(conf), conf["0", "1"], 0)
  
  safe_div <- function(a, b) ifelse(b == 0, 0, a / b)
  
  precision <- safe_div(TP, TP + FP)
  recall <- safe_div(TP, TP + FN)
  f1 <- safe_div(2 * precision * recall, precision + recall)
  accuracy <- mean(y_pred == y_test)
  
  cat("Accuracy :", round(accuracy, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall   :", round(recall, 4), "\n")
  cat("F1 Score :", round(f1, 4), "\n")
  cat("Confusion Matrix:\n")
  print(conf)
  
  if (show_auc) {
    if (!requireNamespace("pROC", quietly = TRUE)) {
      install.packages("pROC")
    }
    library(pROC)
    roc_obj <- roc(y_test, as.numeric(prob_pred))
    cat("AUC      :", round(auc(roc_obj), 4), "\n")
    plot(roc_obj, col = "blue", main = "ROC Curve", legacy.axes = TRUE, xlab = "FPR", ylab = "TPR")
  }
}
evaluation_summary <- evaluate_stan_logistic(fit, X_test, y_test, threshold = 0.5, show_auc = TRUE)


summarize_variable_groups <- function(fit, X_train, group_list) {
  beta_samples <- extract(fit)$beta
  res <- lapply(names(group_list), function(group_name) {
    var_names <- group_list[[group_name]]
    idx <- which(colnames(X_train) %in% var_names)
    
    if (length(idx) == 0) {
      return(data.frame(group = group_name, mean = NA, lower = NA, upper = NA))
    }
    
    if (length(idx) == 1) {
      beta_j <- beta_samples[, idx]
      ci <- quantile(beta_j, c(0.025, 0.975))
      m <- mean(beta_j)
    } else {
      beta_group <- beta_samples[, idx]
      l2 <- apply(beta_group, 1, function(x) sqrt(sum(x^2)))
      ci <- quantile(l2, c(0.025, 0.975))
      m <- mean(l2)
    }
    
    data.frame(group = group_name, mean = m, lower = ci[1], upper = ci[2])
  })
  
  df <- do.call(rbind, res)
  rownames(df) <- NULL
  return(df)
}

group_list <- list(
  Stress = c("VAR_StressDomain_CAT_Red", "VAR_StressDomain_CAT_Blue"),
  Social = c("VAR_SDOHSocialConnectionDomain_CAT_Red", "VAR_SDOHSocialConnectionDomain_CAT_Orange", "VAR_SDOHSocialConnectionDomain_CAT_Blue"),
  Physical = c("VAR_PhysicalActivityDomain_CAT_Red", "VAR_PhysicalActivityDomain_CAT_Orange", "VAR_PhysicalActivityDomain_CAT_Blue"),
  Alcohol = c("VAR_SDOHAlcoholUseDomainRisk_CAT_Red", "VAR_SDOHAlcoholUseDomainRisk_CAT_Blue"), 
  Utilities = c("VAR_UtilitiesDomain_FLG"), 
  Transportation = c("VAR_TransportationDomainCollected_FLG"), 
  Safety = c("VAR_SafetyandDomesticViolenceDomainCollected_FLG"), 
  Housing = c("VAR_HousingStabilityDomainCollected_FLG"), 
  Food_Insecurity = c("VAR_FoodInsecurityDomainCollected_FLG"), 
  Financial_Strain = c("VAR_FinancialResourceStrainDomainCollected_FLG")
)

summary_df <- summarize_variable_groups(fit, X_train, group_list)
print(summary_df)

ggplot(summary_df, aes(x = mean, y = reorder(group, mean))) +
  geom_point() +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Effect size (mean ± 95% CI)", y = NULL,
       title = "Posterior Estimates with 95% Credible Intervals") +
  theme_minimal()



plot_grouped_posterior <- function(fit, X_train, group_list, beta_name = "beta") {
  library(rstan)
  library(ggplot2)
  
  beta_samples <- rstan::extract(fit, pars = beta_name)[[1]]
  
  if (length(dim(beta_samples)) == 3) {
    beta_samples <- beta_samples[, 1, ]
  }
  
  colnames_X <- colnames(X_train)
  
  group_indices <- lapply(group_list, function(vars) {
    which(colnames_X %in% vars)
  })
  
  group_summary <- lapply(group_indices, function(idxs) {
    if (length(idxs) == 0) {
      return(rep(0, nrow(beta_samples)))
    }
    rowSums(beta_samples[, idxs, drop = FALSE])
  })
  
  summary_matrix <- do.call(rbind, lapply(group_summary, function(x) {
    c(mean = mean(x),
      lower = quantile(x, probs = 0.025),
      upper = quantile(x, probs = 0.975))
  }))
  summary_df <- as.data.frame(summary_matrix)
  colnames(summary_df) <- c("mean", "lower", "upper")
  summary_df$Group <- names(group_list)
  summary_df$Group <- factor(summary_df$Group, levels = rev(summary_df$Group))
  
  p <- ggplot(summary_df, aes(x = mean, y = Group)) +
    geom_point() +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Posterior Estimates with 95% Credible Intervals",
         x = "Effect size (mean ± 95% CI)", y = NULL) +
    theme_minimal()
  
  print(p)
  return(invisible(summary_df))
}

plot_selected_posterior <- function(fit, X_train, selected_vars, beta_name = "beta") {
  library(rstan)
  library(ggplot2)
  
  beta_samples <- rstan::extract(fit, pars = beta_name)[[1]]
  
  if (length(dim(beta_samples)) == 3) {
    beta_samples <- beta_samples[, 1, ]
  }
  
  colnames_X <- colnames(X_train)
  
  selected_idx <- which(colnames_X %in% selected_vars)
  
  if (length(selected_idx) == 0) {
    stop("None of the selected variables were found in X_train.")
  }
  
  beta_selected <- beta_samples[, selected_idx, drop = FALSE]
  var_names <- colnames_X[selected_idx]
  
  summary_matrix <- apply(beta_selected, 2, function(x) {
    c(mean = mean(x),
      lower = quantile(x, 0.025),
      upper = quantile(x, 0.975))
  })
  summary_df <- as.data.frame(t(summary_matrix))
  colnames(summary_df) <- c("mean", "lower", "upper")
  summary_df$Variable <- var_names
  
  summary_df$Variable <- factor(summary_df$Variable, levels = rev(summary_df$Variable))
  
  p <- ggplot(summary_df, aes(x = mean, y = Variable)) +
    geom_point() +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Posterior Estimates for Selected Variables",
         x = "Effect size (mean ± 95% CI)", y = NULL) +
    theme_minimal()
  
  print(p)
  return(invisible(summary_df))
}

posterior_summary <- plot_grouped_posterior(fit, X_train, group_list)

selected_vars <- c("VAR_StressDomain_CAT_Red", "VAR_StressDomain_CAT_Blue", 
                   "VAR_SDOHSocialConnectionDomain_CAT_Red", "VAR_SDOHSocialConnectionDomain_CAT_Orange", "VAR_SDOHSocialConnectionDomain_CAT_Blue", 
                   "VAR_PhysicalActivityDomain_CAT_Red", "VAR_PhysicalActivityDomain_CAT_Orange", "VAR_PhysicalActivityDomain_CAT_Blue", 
                   "VAR_SDOHAlcoholUseDomainRisk_CAT_Red", "VAR_SDOHAlcoholUseDomainRisk_CAT_Blue")

selected_vars <- c("VAR_UtilitiesDomain_FLG", 
                   "VAR_TransportationDomainCollected_FLG", 
                   "VAR_SafetyandDomesticViolenceDomainCollected_FLG", 
                   "VAR_HousingStabilityDomainCollected_FLG", 
                   "VAR_FoodInsecurityDomainCollected_FLG", 
                   "VAR_FinancialResourceStrainDomainCollected_FLG")

select_credible_interval <- plot_selected_posterior(fit, X_train, selected_vars)

summarize_beta <- function(fit, X_train, beta_name = "beta") {
  beta_samples <- rstan::extract(fit, pars = beta_name)[[1]]
  
  if (length(dim(beta_samples)) == 3) {
    beta_samples <- beta_samples[, 1, ]
  }
  
  beta_means <- apply(beta_samples, 2, mean)
  beta_vars  <- apply(beta_samples, 2, var)
  beta_lower <- apply(beta_samples, 2, function(x) quantile(x, 0.025))
  beta_upper <- apply(beta_samples, 2, function(x) quantile(x, 0.975))
  
  variable_names <- colnames(X_train)
  
  if (length(variable_names) != length(beta_means)) {
    stop("Number of variable names does not match number of beta coefficients.")
  }
  
  beta_df <- data.frame(
    Variable = variable_names,
    Mean = beta_means,
    Variance = beta_vars,
    Lower = beta_lower,
    Upper = beta_upper
  )
  
  return(beta_df)
}
beta_summary <- summarize_beta(fit, X_train)
print(beta_summary)


write.csv(X_train, "X_train_SDOH.csv", row.names = FALSE)
write.csv(X_test, "X_test_SDOH.csv", row.names = FALSE)
write.csv(y_train, "y_train_SDOH.csv", row.names = FALSE)
write.csv(y_test, "y_test_SDOH.csv", row.names = FALSE)
