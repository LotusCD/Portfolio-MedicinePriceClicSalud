### Analysis Report on Medication Pricing Factors

**Note**: This is a learning project and is not intended for use in real decision-making scenarios. As a learning project, it is open to improvements and enhancements. Further iterations and refinements can help increase the robustness and applicability of the findings.


### Analysis Report on Medication Pricing Factors

#### 1. Introduction
This report presents an analysis of the factors influencing the cost of medications. The study utilizes data on various attributes, including the unit of dispensation, active ingredients, manufacturers, concentration, and distribution channels. The analysis is conducted using XGBoost regression to determine the importance of these features in predicting medication prices. The data used in this study is sourced from the [Clicsalud - Termómetro de Precios de Medicamentos](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Clicsalud-Term-metro-de-Precios-de-Medicamentos/n4dj-8r7k/about_data).


#### 2. Understanding Log Prices
In this analysis, prices are referred to as "log prices" because the logarithm of the actual prices has been used. Transforming prices using the natural logarithm helps stabilize variance and make the distribution more normal-like. This transformation is particularly useful in reducing the impact of outliers and skewed data, enabling more accurate and reliable statistical analysis and modeling.

#### 3. Average Cost by Unidad de Dispensacion
The analysis of average costs by unit of dispensation reveals significant variability:

- **Highest Average Costs**: The units of dispensation with the highest average log prices are "Emulsión Oral," "Implante," and "Gas." These forms likely incur higher production or packaging costs, contributing to their elevated prices.
- **Overall Trend**: The variability in costs indicates that the form in which medication is dispensed plays a crucial role in its pricing. This can be attributed to differences in manufacturing processes, material requirements, and delivery mechanisms.

![Average Cost by Unidad de Dispensacion](AverageCostbyUnidaddeDispensacion.png)
  

#### 4. Average Cost by Principio Activo
The average costs associated with different active ingredients also show substantial differences:

- **Highest Average Costs**: "mometasona," "nitazoxanida," and "oximetazolina" are the active ingredients with the highest average log prices. These differences may result from the rarity of the ingredients, their therapeutic importance, or higher production costs.
- **Cost Variation**: The significant variation in costs among different active ingredients suggests that the choice of active ingredient is a key determinant of medication pricing.

![Average Cost by Principio Activo](AverageCostbyPrincipioActivo.png)

#### 5. Average Cost by Fabricante
An analysis of average costs by manufacturer indicates a wide range of pricing:

- **Highest Average Costs**: Manufacturers such as "Sophia," "Sophia Beta," and "Salus Pharma" have the highest average log prices. Factors such as brand reputation, production capabilities, and market strategies likely influence these costs.
- **Manufacturer Influence**: The variability in costs among manufacturers underscores the significant impact of the producer on medication pricing, potentially due to differences in manufacturing efficiency, quality, and brand value.

![Average Cost by Fabricante](AverageCostbyFabricante.png)

#### 6. Feature Importance from XGBoost Model
The XGBoost regression model identifies the relative importance of different features in predicting medication prices:

- **Most Important Features**: The unit of dispensation ("unidad_de_dispensacion_encoded") is the most important predictor of medication price. This is followed by the active ingredient ("principio_activo_encoded") and the manufacturer ("fabricante_encoded").
- **Moderate Importance**: The concentration of the medication ("concentracion_en_gramos") and the factor number ("numerofactor") have moderate importance.
- **Least Important**: The distribution channel ("canal_encoded") has the least impact on pricing.

![Feature Importance](Feature%20Importance%20from%20Random%20Forest.png)

#### 7. Model Performance Comparison
A comparison of model performance metrics across different regression models highlights the effectiveness of XGBoost:

| Model             | MAE      | RMSE     | MSE      | R^2       |
|-------------------|----------|----------|----------|-----------|
| Random Forest     | 0.9165   | 1.3330   | 1.7768   | 0.6524    |
| Gradient Boosting | 1.1391   | 1.5617   | 2.4388   | 0.5229    |
| **XGBoost**       | **0.8783** | **1.2574** | **1.5812** | **0.6907** |
| LightGBM          | 0.9578   | 1.3388   | 1.7924   | 0.6494    |
| CatBoost          | 0.9919   | 1.3719   | 1.8821   | 0.6318    |
| SVR               | 1.6848   | 2.2035   | 4.8555   | 0.0501    |

- **XGBoost Performance**: XGBoost outperforms other models, achieving the lowest Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), as well as the highest R-squared (R^2) value. This indicates that XGBoost provides the most accurate predictions for medication prices among the models tested.

#### 8. Conclusion
The analysis demonstrates that the form of medication, active ingredients, and manufacturers are the primary factors influencing medication prices. The unit of dispensation is the most significant predictor, reflecting the impact of production and packaging processes on costs. Active ingredients and manufacturers also play crucial roles, highlighting the importance of ingredient sourcing and brand reputation. The concentration and distribution channels have lesser but still notable effects on pricing.

XGBoost has been identified as the most effective model for predicting medication prices, providing superior accuracy compared to other models. These insights provide a comprehensive understanding of the factors affecting medication costs and can inform decision-making in the pharmaceutical industry, from production planning to pricing strategies.

**Note**: This is a learning project and is not intended for use in real decision-making scenarios. As a learning project, it is open to improvements and enhancements. Further iterations and refinements can help increase the robustness and applicability of the findings.
