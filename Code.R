# Data manipulations
library(dplyr)
library(tidyr)
library(lubridate)
library(zoo)
library(Metrics) # Evaluation metrics
library(psych) # Descriptive statistics
library(ranger) # Random Forest
library(caret) # Grid search
library(glmnet) # Elastic Net
library(nnet) # MLP
library(kknn) # k-NN
library(xgboost) # XGBoost
library(kernlab) # SVM

# ---------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

r_squared <- function(actual, predicted) {
  # Calculate R^2
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  1 - (ss_res / ss_tot)
}

evaluate_model <- function(actual, predicted) {
  # Evaluate the model with RMSE, MAE and R^2
  cat("RMSE:", rmse(actual, predicted), "\n")
  cat("MAE:", mae(actual, predicted), "\n")
  cat("R^2:", r_squared(actual, predicted), "\n")
}

grid_search_train <- function(model, grid, train_data, formula) {
  # Grid search for the best model over parameters in the grid
  train_control <- trainControl(
    method = "cv", 
    number = 5
  )
  # Some models need specific parameters
  if (model == "nnet") {
    model <- caret::train(
      formula,
      data = train_data,
      method = model,
      trControl = train_control,
      tuneGrid = grid,
      linout = TRUE,
      maxit = 500,
      MaxNWts = 10000,
      metric = "RMSE"
    )
  }
  else if (model == "svmRadial") {
    model <- caret::train(
      formula,
      data = train_data,
      method = model,
      trControl = train_control,
      tuneGrid = grid,
      type="nu-svr",
      metric = "RMSE"
    )
  }
  else {
    model <- caret::train(
      formula,
      data = train_data,
      method = model,
      trControl = train_control,
      tuneGrid = grid,
      metric = "RMSE"
    )
  }
  
  return(model)
}

preprocess_poly <- function(df, degree, numeric_cols_idx) {
  # Create polynomial terms (x,x^2,x^3,...,x^k) and add them to the dataframe
  df_poly <- df
  for (idx in numeric_cols_idx) {
    col_name <- colnames(df)[idx]
    poly_terms <- poly(df[[col_name]], degree = degree, raw = TRUE)
    
    # Rename columns and bind them
    colnames(poly_terms) <- paste0(col_name, "_poly_", 1:degree)
    df_poly <- cbind(df_poly, poly_terms)
  }
  
  # Drop original numeric columns
  df_poly <- df_poly[, -numeric_cols_idx, drop = FALSE]
  
  return(df_poly)
}

grid_search_poly <- function(model, grid, train_data, degree, numeric_cols_idx) {
  # Grid search for the best model over polynomial, interaction terms and parameters of the model in the grid
  best_model <- NULL
  best_rmse <- NULL
  best_degree <- 0
  interaction <- FALSE

  # Base model without poly and interaction terms
  formula <- as.formula(paste("destroyed_not_reached  ~", paste(colnames(train_data[,-which(colnames(train_data) == "destroyed_not_reached")]), collapse = " + ")))
  model_without_interaction <- grid_search_train(model, grid, train_data, formula)
  best_rmse <- min(model_without_interaction$results$RMSE)
  best_model <- model_without_interaction

  # No poly with interaction terms
  formula_with_int <- as.formula(paste("destroyed_not_reached  ~ (", paste(colnames(train_data[,-which(colnames(train_data) == "destroyed_not_reached")]), collapse = " + "), ")^2"))
  model_with_interaction <- grid_search_train(model, grid, train_data, formula_with_int)
  if (min(model_with_interaction$results$RMSE) < best_rmse) {
    best_rmse <- min(model_with_interaction$results$RMSE)
    best_model <- model_with_interaction
    interaction <- TRUE
  }

  for (d in 2:degree) {
    # Create polynomial terms
    train_data_poly <- preprocess_poly(train_data, d, numeric_cols_idx)

    # Train model with polynomial terms only
    formula_poly <- as.formula(paste("destroyed_not_reached  ~", paste(colnames(train_data_poly[,-which(colnames(train_data_poly) == "destroyed_not_reached")]), collapse = " + ")))
    model_poly <- grid_search_train(model, grid, train_data_poly, formula_poly)
    if (min(model_poly$results$RMSE) < best_rmse) {
      best_rmse <- min(model_poly$results$RMSE)
      best_model <- model_poly
      best_degree <- d
    }

    # Train model with polynomial terms and interaction terms
    formula_with_int <- as.formula(paste("destroyed_not_reached  ~ (", paste(colnames(train_data_poly[,-which(colnames(train_data_poly) == "destroyed_not_reached")]), collapse = " + "), ")^2"))
    model_poly_int <- grid_search_train(model, grid, train_data_poly, formula_with_int)
    if (min(model_poly_int$results$RMSE) < best_rmse) {
      best_rmse <- min(model_poly_int$results$RMSE)
      best_model <- model_poly_int
      best_degree <- d
      interaction <- TRUE
    }
  }

  return(list(best_model, best_rmse, best_degree, interaction))
}

evaluate_poly_grid_search <- function(grid_poly_object, train_data, test_data, numeric_cols_idx) {
  # Evaluate the best model found by grid search over polynomial and interaction terms
  if (grid_poly_object[[3]] != 0) {
    train_data_poly <- preprocess_poly(train_data, grid_poly_object[[3]], numeric_cols_idx)
    test_data_poly <- preprocess_poly(test_data, grid_poly_object[[3]], numeric_cols_idx)
    
    print("Training:")
    evaluate_model(train_data$destroyed_not_reached, predict(grid_poly_object[[1]], train_data_poly))
    print("Testing:")
    evaluate_model(test_data$destroyed_not_reached, predict(grid_poly_object[[1]], test_data_poly))
  }
  else {
    print("Training:")
    evaluate_model(train_data$destroyed_not_reached, predict(grid_poly_object[[1]], train_data))
    print("Testing:")
    evaluate_model(test_data$destroyed_not_reached, predict(grid_poly_object[[1]], test_data))
  }
}

grid_seach_knn <- function(grid, train_data, test_data, formula, cv_k) {
  # Grid search for the best k-NN model over parameters in the grid
  best_model <- NULL
  best_model_idx <- NULL
  best_rmse <- Inf
  
  for (i in 1:nrow(grid)) {
    params <- grid[i, ]
    
    # Perform cross-validation
    folds <- createFolds(train_data$destroyed_not_reached, k = cv_k, list = TRUE)
    rmse_values <- sapply(folds, function(fold) {
      train_fold <- train_data[-fold, ]
      test_fold <- train_data[fold, ]
      
      model_fold <- kknn(
        formula,
        train = train_fold,
        test = test_fold,
        k = params$k,
        distance = params$distance,
        kernel = as.character(params$kernel),
        scale = params$scale
      )
      
      predictions <- fitted(model_fold)
      rmse(test_fold$destroyed_not_reached, predictions)
    })
    
    mean_rmse <- mean(rmse_values)
    
    if (mean_rmse < best_rmse) {
      best_rmse <- mean_rmse
      best_model_idx <- i
    }
  }
  
  best_params <- grid[best_model_idx, ]
  best_model <- kknn(
    formula,
    train = train_data,
    test = test_data,
    k = best_params$k,
    distance = best_params$distance,
    kernel = as.character(best_params$kernel),
    scale = best_params$scale
  )
  
  return(list(best_model = best_model, best_rmse = best_rmse, params = best_params))
}

ensemble_model <- function(models, data, weights = NULL) {
  # Calculate the predictions of the ensemble model
  # If weights are not provided, use equal weights 
  if (is.null(weights)) {
    weights <- rep(1/length(models), length(models))
  }
  # Otherwise, we use 1/Metric (e.g. RMSE) as weights
  else {
    weights <- (1/weights)/sum(1/weights)
  }
  
  # Assume that the first model is GLMNET
  predictions <- sapply(models[2:length(models)], function(model) predict(model, data))

  test_data_poly <- preprocess_poly(data, models[[1]][[3]], NUMERIC_COLS_IDX)
  prediction_glm <- predict(models[[1]][[1]], test_data_poly)
  weighted_predictions <- cbind(prediction_glm, predictions) %*% weights

  return(weighted_predictions)
}

permutation_test_rmse <- function(predictions1, predictions2, target, n_permutations = 1000) {
  # Perform a permutation test to determine if the RMSE of two models is significantly different
  rmse1 <- rmse(target, predictions1)
  rmse2 <- rmse(target, predictions2)
  
  perm_rmse <- numeric(n_permutations)
  
  for (i in 1:n_permutations) {
    # Permute the target variable
    model_to_choose <- sample(1:2, length(target), replace = TRUE)
    permuted_target <- ifelse(model_to_choose == 1, predictions1, predictions2)

    # Calculate the RMSE and store it
    perm_rmse[i] <- rmse(target, permuted_target)
  }
  
  # Calculate the p-value
  p_value_1 <- mean(perm_rmse >= rmse1)
  p_value_2 <- mean(perm_rmse >= rmse2)
  p_value <- min(p_value_1, p_value_2)
  
  return(p_value = p_value)
}

# ---------------------------------------------------------------------------------------------------------------------
# READ DATA
# ---------------------------------------------------------------------------------------------------------------------

data <- read.csv("missile_attacks_daily.csv", encoding = "UTF-8")
weather <- read.csv("export.csv")


# ---------------------------------------------------------------------------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------------------------------------------------------------------------

#Remove unnecessary columns and fill NAs
data <- data[, -c(2,5,10,11,12,13,14,15,16)]
data[data == ""] <- NA

# Create one-hot encoded carrier variables
print(count(data, carrier, name = "count"))
data <- data %>%
  mutate(
    `Bomber_Carrier` = as.integer(grepl("Tu", carrier, ignore.case = TRUE)),
    `Jet_Carrier` = as.integer(grepl("MiG|Su", carrier, ignore.case = TRUE)),
    `Naval_Carrier` = as.integer(grepl("Navi|Admiral|Submarines|Novorossiysk", carrier, ignore.case = TRUE)),
    `Missile_System` = as.integer(grepl("Bastion|Iskander", carrier, ignore.case = TRUE)),
    `NA_Carrier` = as.integer(is.na(carrier))
  )

# Drone/Rocket model one-hot encoded columns
print(count(data, model, name = "count"))
data <- data %>%
  mutate(
    # Ballistic Missiles
    `Iskander_Ballistic_Missile` = as.integer(grepl("Iskander", model, ignore.case=TRUE)),
    `Kinzhal_Ballistic_Missile` = as.integer(grepl("Kinzhal", model, ignore.case=TRUE)),
    `Ballistic_Missile` = as.integer(grepl("Intercontinental|KN-23|Kinzhal|Iskander", model, ignore.case=TRUE)),
    # Cruise Missiles
    `Kalibr_Cruise_Missile` = as.integer(grepl("Kalibr", model, ignore.case=TRUE)),
    `X101_Cruise_Missile` = as.integer(grepl("X-101", model, ignore.case=TRUE)),
    `X59_Cruise_Missile` = as.integer(grepl("X-59", model, ignore.case=TRUE)),
    `X31_Cruise_Missile` = as.integer(grepl("X-31", model, ignore.case=TRUE)),
    `X22_Cruise_Missile` = as.integer(grepl("X-22", model, ignore.case=TRUE)),
    `Cruise_Missile` = as.integer(grepl("Oniks|Kalibr|Zircon|X-101|X-69|X-59|X-35|X-32|X-31|X-22", model, ignore.case=TRUE)),
    # Anti-Aircraft
    `C300_Aircraft_Missile` = as.integer(grepl("C-300", model, ignore.case=TRUE)),
    `C400_Aircraft_Missile` = as.integer(grepl("C-400", model, ignore.case=TRUE)),
    # Drones
    `Shahed_Drone` = as.integer(grepl("Shahed", model, ignore.case=TRUE)),
    `Orlan_Drone` = as.integer(grepl("Orlan", model, ignore.case=TRUE)),
    `Reconnaissance_Drone` = as.integer(grepl("Reconnaissance", model, ignore.case=TRUE)),
    `Supercam_Drone` = as.integer(grepl("Supercam", model, ignore.case=TRUE)),
    `Lancet_Drone` = as.integer(grepl("Lancet", model, ignore.case=TRUE)),
    `Merlin_Drone` = as.integer(grepl("Merlin", model, ignore.case=TRUE)),
    `Zala_Drone` = as.integer(grepl("ZALA", model, ignore.case=TRUE)),
    `Drone` = as.integer(grepl("Shahed|Orlan|Reconnaissance|Supercam|Lancet|Merlin|ZALA|Eleron|Forpost|Granat|Mohajer|Orion|UAV|Картограф|Молнія|Привет-82|Фенікс", model, ignore.case=TRUE)),
    # Other Missiles
    `Other_Missile` = as.integer(grepl("Aerial|KAB|Kub|Unknown Missile", model, ignore.case=TRUE)),
  )

# Infer the carrier from the rocket type
for (i in 1:nrow(data)) {
  if (data$Iskander_Ballistic_Missile[i] | data$C300_Aircraft_Missile[i] | data$C400_Aircraft_Missile[i]) {
    data$Missile_System[i] <- as.integer(1)
    data$NA_Carrier[i] <- as.integer(0)
  } else if (data$Kalibr_Cruise_Missile[i]) {
    data$Naval_Carrier[i] <- as.integer(1)
    data$NA_Carrier[i] <- as.integer(0)
  } else if (data$X59_Cruise_Missile[i] | data$X31_Cruise_Missile[i]){
    data$Jet_Carrier[i] <- as.integer(1)
    data$NA_Carrier[i] <- as.integer(0)
  } else if (data$X101_Cruise_Missile[i] | data$X22_Cruise_Missile[i]){
    data$Bomber_Carrier[i] <- as.integer(1)
    data$NA_Carrier[i] <- as.integer(0)
  }
}

# Launch locations one-hot encoded columns
print(count(data, launch_place, name = "count"))
data <- data %>%
  mutate(
    # South
    `Crimea_Location` = as.integer(grepl("Crimea", launch_place, ignore.case=TRUE)),
    `Black_Sea_Location` = as.integer(grepl("Black Sea", launch_place, ignore.case=TRUE)),
    `Primorsko_Akhtarsk_Location` = as.integer(grepl("Primorsko-Akhtarsk", launch_place, ignore.case=TRUE)),
    `Yeysk_Location` = as.integer(grepl("Yeysk", launch_place, ignore.case=TRUE)),
    `South_Location` = as.integer(grepl("south|Black Sea|Crimea|Tokmak|Sea of Azov|Krasnodar|Yeysk|Kherson|Zaporizhzhia|Melitopol|Primorsko-Akhtarsk|Berdiansk|Taganrog", launch_place, ignore.case=TRUE)),
    # East
    `Millerovo_Location` = as.integer(grepl("Millerovo", launch_place, ignore.case=TRUE)),
    `Voronezh_Location` = as.integer(grepl("Voronezh", launch_place, ignore.case=TRUE)),
    `East_Location` = as.integer(grepl("east|Donetsk|Astrakhan|Voronezh|Krasnodar|Millerovo|Saratov|Volgodonsk|Yeysk|Rostov|Engels-2|Lipetsk|Luhansk|Morozovsk|Primorsko-Akhtarsk|Ryazan|Tambov|Savasleyka|Taganrog|Tula|Volgograd", launch_place, ignore.case=TRUE)),
    # North
    `Belgorod_Location` = as.integer(grepl("Belgorod", launch_place, ignore.case=TRUE)),
    `Kursk_Location` = as.integer(grepl("Kursk", launch_place, ignore.case=TRUE)),
    `Bryansk_Location` = as.integer(grepl("Bryansk", launch_place, ignore.case=TRUE)),
    `North_Location` = as.integer(grepl("north|Belgorod|Kursk|Bryansk|Belarus|Voronezh|Oryol|Olenya|Lipetsk|Olenegorsk|Ryazan|Tambov|Savasleyka|Soltsy-2|Shaykovka|Tula", launch_place, ignore.case=TRUE)),
    # Ukraine
    `Zaporizhzhia_Location` = as.integer(grepl("Zaporizhzhia|Tokmak|Melitopol|Berdiansk", launch_place, ignore.case=TRUE)),
    `Ukraine_Location` = as.integer(grepl("Donetsk|Tokmak|Kherson|Zaporizhzhia|Luhansk|Melitopol|Berdiansk", launch_place, ignore.case=TRUE)),
    # Deep inside Russia
    `Caspian_Sea_Location` = as.integer(grepl("Caspian", launch_place, ignore.case=TRUE)),
    `Oryol_Location` = as.integer(grepl("Oryol", launch_place, ignore.case=TRUE)),
    `Deep_Russia_Location`= as.integer(grepl("Caspian|Astrakhan|Voronezh|Oryol|Saratov|Engels-2|Olenya|Lipetsk|Olenegorsk|Ryazan|Tambov|Savasleyka|Soltsy-2|Shaykovka|Tula", launch_place, ignore.case=TRUE)),
    #Unknown
    `Unknown_Location` = as.integer(is.na(launch_place))
  )

# Time variables
data <- data %>%
  mutate(
    # Transform the column to Date type, removing any time component
    time_start = as.Date(substr(time_start, 1, 10)),
    
    # Create Weekday column (e.g., Monday, Tuesday, etc.)
    Weekday = weekdays(time_start),
    
    # Create Season column based on the month
    Season = case_when(
      month(time_start) %in% c(12, 1, 2) ~ "Winter",
      month(time_start) %in% c(3, 4, 5) ~ "Spring",
      month(time_start) %in% c(6, 7, 8) ~ "Summer",
      month(time_start) %in% c(9, 10, 11) ~ "Fall"
    )
  )

# Convert to factor
data[, c("Weekday", "Season")] <- lapply(data[, c("Weekday", "Season")], as.factor)
# Create dummy variables for Weekday and Season
dummy_vars_encoder <- dummyVars(~ Weekday + Season, data = data)
data <- cbind(data, predict(dummy_vars_encoder, newdata = data))
# Remove original columns and one of the dummy columns to avoid multicollinearity
data <- data %>% select(-Weekday, -Season, -Weekday.Friday, -Season.Winter)

# Calculate the sum of destroyed and not_reach_goal columns 
data <- data %>%
  # Remove observations for which launched column has NAs (only 3 obs)
  filter(!is.na(launched)) %>%
  # Add destroyed and not_reach_goal columns together
  mutate(destroyed_not_reached = destroyed + coalesce(not_reach_goal, 0)) %>%
  # Remove wrong observations (only 1 obs)
  filter(destroyed_not_reached <= launched)

# Calculate moving average destruction rate
data$destruction_rate <- data$destroyed_not_reached/data$launched
data <- data %>%
  # Arrange by date (oldest to newest)
  arrange(time_start) %>%
  # Add MAs
  # WE COULD ALSO TUNE FOR THE BEST LAGS
  mutate(
    `MA30_destruction_rate` = lag(rollmean(destruction_rate, k = 30, fill = NA, align = "right")),
    `MA7_destruction_rate` = lag(rollmean(destruction_rate, k = 7, fill = NA, align = "right")),
    `MA100_destruction_rate` = lag(rollmean(destruction_rate, k = 100, fill = NA, align = "right")),
  )
  
# MERGE WITH WEATHER DATA

weather[["tsun"]] <- NULL
# Convert all NAs to 0s
weather[is.na(weather)] <- 0

weather$date = as.Date(weather$date, format = "%m/%d/%Y")

data <- left_join(data, weather, by=c("time_start" = "date"))

# Remove unnecessary columns
data[["carrier"]] <- NULL
data[["model"]] <- NULL
data[["launch_place"]] <- NULL
data[["destroyed"]] <- NULL
data[["not_reach_goal"]] <- NULL
data[["destruction_rate"]] <- NULL
data[["time_start"]] <- NULL
data[["Kalibr_Cruise_Missile"]] <- NULL # It is perfectly correlated with Naval_Carrier

# To avoid multicollinearity
data[["NA_Carrier"]] <- NULL
data[["Unknown_Location"]] <- NULL
data[["Other_Missile"]] <- NULL
data[["tmax"]] <- NULL
data[["tmin"]] <- NULL

# Omit NAs that appeared after creating MAs
data <- na.omit(data)

# ---------------------------------------------------------------------------------------------------------------------
# DATA SUMMARY
# ---------------------------------------------------------------------------------------------------------------------

str(data)
summary(data)
describe(data)

cor_matrix <- cor(data)
summary(cor_matrix[upper.tri(cor_matrix)])

highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.9, names = TRUE)
print(highly_correlated)

# ---------------------------------------------------------------------------------------------------------------------
# BASELINE - We just take average destruction rate and then infer number of destructed from number of launched 
# ---------------------------------------------------------------------------------------------------------------------

# Determine the indices for the training set
set.seed(131213123) 
train_indices <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
# Split the data
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]


avg_destruction_rate <- mean(train_data$destroyed_not_reached/train_data$launched)
# RMSE ~4.7 - 5
# MAE ~2.3
# R^2 ~0.938
evaluate_model(test_data$destroyed_not_reached, test_data$launched*avg_destruction_rate)

ols_simple <- lm(destroyed_not_reached ~ launched, data=train_data)
# RMSE ~2.3
# MAE ~1.5
# R^2 ~0.9856686
evaluate_model(test_data$destroyed_not_reached, predict(ols_simple, test_data))

# ---------------------------------------------------------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------------------------------------------------------

# Create formula string
FORMULA <- as.formula(paste("destroyed_not_reached  ~", paste(colnames(data[,-which(colnames(data) == "destroyed_not_reached")]), collapse = " + ")))
# Determine the indices for the target, binary and numeric columns
TARGET_IDX <- which(colnames(data) == "destroyed_not_reached")
BINARY_COLS_IDX <- which(sapply(data, function(col) length(unique(col)) == 2))
NUMERIC_COLS_IDX <- which(!colnames(data) %in% colnames(data)[c(TARGET_IDX, BINARY_COLS_IDX)])

# Standardize numeric columns
scaler <- preProcess(train_data[, NUMERIC_COLS_IDX], method = c("center", "scale"))
train_data[, NUMERIC_COLS_IDX] <- predict(scaler, train_data[, NUMERIC_COLS_IDX])
test_data[, NUMERIC_COLS_IDX] <- predict(scaler, test_data[, NUMERIC_COLS_IDX])


# OLS
ols <- lm(FORMULA, data=train_data)

# TEST RMSE ~2.10, MAE ~1.22, R^2 ~0.9887
evaluate_model(train_data$destroyed_not_reached, ols$fitted.values)
evaluate_model(test_data$destroyed_not_reached, predict(ols, test_data)) 


# ELASTIC NET (It is a linear combination of L1 (Lasso) and L2 (Ridge) regularization)
glmnet_grid <- expand.grid(
  alpha = seq(0,1, by=0.1), # 0 - Ridge, 1 - Lasso
  lambda = 10^seq(-5, 5, by = 0.1)
)

# This searches over different polynomials and interactions in addition to parameters of Elastic Net
glmnet_model_object <- grid_search_poly("glmnet", glmnet_grid, train_data, degree=5, NUMERIC_COLS_IDX)
# Best combination is poly_degree = 4, interaction = FALSE, alpha = 0.4, lambda = 0.07943282
# TEST RMSE ~1.967, MAE ~1.15, R^2 ~0.99
evaluate_poly_grid_search(glmnet_model_object, train_data, test_data, NUMERIC_COLS_IDX)


# RANDOM FOREST
forest_grid <- expand.grid(
  mtry = c(59),
  splitrule = c('variance'),
  min.node.size = c(1,5,10,20,50,100)
)

forest_model <- grid_search_train("ranger", forest_grid, train_data, FORMULA)
evaluate_model(train_data$destroyed_not_reached, predict(forest_model, train_data))
# Best combination is num.trees = 500, mtry = 59, splitrule="variance", min.node.size=1
# TEST RMSE ~2.07, MAE ~0.989, R^2 ~0.99
evaluate_model(test_data$destroyed_not_reached, predict(forest_model, test_data))


# XGBoosted trees
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 500),
  max_depth = c(2, 3, 4, 5, 7),
  eta = c(0.01, 0.1, 0.15, 0.3, 1),
  gamma = c(0, 1, 0.5),
  colsample_bytree = c(0.5, 0.9, 1),
  min_child_weight = c(0.5, 1, 3),
  subsample = c(0.5, 0.8, 1)
)

xgb_model <- grid_search_train("xgbTree", xgb_grid, train_data, FORMULA)
evaluate_model(train_data$destroyed_not_reached, predict(xgb_model, train_data))
# Best combination is num.trees=100, max_depth=3, eta=0.1, gamma=0.5, colsample_bytree = 1, min_child_weight = 1 and subsample= 0.8.
# TEST RMSE ~1.914, MAE ~1.06, R^2 ~0.99
evaluate_model(test_data$destroyed_not_reached, predict(xgb_model, test_data))

# MLP
mlp_grid <- expand.grid(
  size = c(100),
  decay = c(0, 10^seq(0, 2, by = 0.1))
)

mlp_model <- grid_search_train("nnet", mlp_grid, train_data, FORMULA)
evaluate_model(train_data$destroyed_not_reached, predict(mlp_model, train_data))
# Best combination is size=100, decay=15.8.
# TEST RMSE ~2.18, MAE ~1.22, R^2 ~0.987
# LIKELY MODEL IS NOT TUNED PROPERLY OR ISSUES WITH PACKAGE FUNCTION
evaluate_model(test_data$destroyed_not_reached, predict(mlp_model, test_data))

# kNN
knn <- expand.grid(
  k = seq(5,100, by=5),
  distance = 1:5,
  kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"),
  scale = c(TRUE, FALSE)
)

knn_model <- grid_seach_knn(knn, train_data, test_data, FORMULA, cv_k = 5)
# Best combination is k=10, distance=4, kernel=epanechnikov, scale=FALSE
# TEST RMSE ~5.15, MAE ~3.02, R^2 ~0.93 BAAAAAD
evaluate_model(test_data$destroyed_not_reached, fitted(knn_model[[1]]))

# SVM
svm_grid <- expand.grid(
  sigma = 2^seq(-15, -2, by = 2),
  C = c(100, 500, 1000, 2000)
)

svm_model <- grid_search_train("svmRadial", svm_grid, train_data, FORMULA)
evaluate_model(train_data$destroyed_not_reached, predict(svm_model, train_data))
# Best combination is sigma = 0.0001220703 and C = 500.
# TEST RMSE ~2.08, MAE ~1.25, R^2 ~0.989 
# LIKELY MODEL IS NOT TUNED PROPERLY OR ISSUES WITH PACKAGE FUNCTION
evaluate_model(test_data$destroyed_not_reached, predict(svm_model, test_data))


# ENSEMBLE MODELS

models <- list(glmnet_model_object, ols, forest_model, xgb_model, mlp_model, svm_model)
weights <- c(1.967,2.1,2.07,1.914,2.18,2.08) # RMSEs

# UNWEIGHTED ALL
ensemble_predictions_test <- ensemble_model(models, test_data)
# TEST RMSE ~1.847532, MAE ~1.016913 , R^2 ~ 0.9913312
evaluate_model(test_data$destroyed_not_reached, ensemble_predictions_test)

# WEIGHTED ALL
ensemble_predictions_test_weighted <- ensemble_model(models, test_data, weights)
# TEST RMSE ~1.846386, MAE ~1.01628, R^2 ~ 0.9913419
evaluate_model(test_data$destroyed_not_reached, ensemble_predictions_test_weighted)

# WEIGHTED BEST
models_best <- list(glmnet_model_object, forest_model, xgb_model)
weights_best <- c(1.967, 2.07, 1.914)
ensemble_predictions_test_best <- ensemble_model(models_best, test_data, weights_best)
# TEST RMSE ~1.856588, MAE ~1.00893, R^2 ~ 0.991246
evaluate_model(test_data$destroyed_not_reached, ensemble_predictions_test_best)


# PERMUTATION TEST - TEST IF MODELS ARE SIGNIFICANTLY DIFFERENT

# Generate predictions
glm_predictions <- predict(glmnet_model_object[[1]], preprocess_poly(test_data, glmnet_model_object[[3]], NUMERIC_COLS_IDX))
ols_predictions <- predict(ols, test_data)
xgb_predictions <- predict(xgb_model, test_data)
forest_predictions <- predict(forest_model, test_data)
svm_predictions <- predict(svm_model, test_data)

# Compare all models with OLS
# P-values are after each line. They are the p-values that one model is better than the other
p_value_glm_ols <- permutation_test_rmse(glm_predictions, ols_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.00098
p_value_xgb_ols <- permutation_test_rmse(xgb_predictions, ols_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.01564
p_value_rf_ols <- permutation_test_rmse(forest_predictions, ols_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.441 - no-difference
p_value_svm_ols <- permutation_test_rmse(svm_predictions, ols_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.48022 - no-difference
p_value_ensemble_ols <- permutation_test_rmse(ensemble_predictions_test, ols_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0

# Compare best models with each other
p_value_glm_xgb <- permutation_test_rmse(glm_predictions, xgb_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.24546 - no-difference
p_value_ensemble_xgb <- permutation_test_rmse(ensemble_predictions_test, xgb_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.14074 - no-difference
p_value_ensemble_glm <- permutation_test_rmse(ensemble_predictions_test, glm_predictions, test_data$destroyed_not_reached, n_permutations=50000) # 0.00218

p_value_glm_ols
p_value_xgb_ols
p_value_rf_ols
p_value_svm_ols
p_value_ensemble_ols
p_value_glm_xgb
p_value_ensemble_xgb
p_value_ensemble_glm
