# burnt_labels
## Executive Summary
### Project Overview
This project leverages Large Language Models (LLMs) to enhance product categorization for efficient inventory management in the restaurant industry. Focusing on automating back-of-house operations, the project addresses the challenge of standardizing product descriptions from multiple suppliers, which is crucial for efficient inventory management which leads to minimizing waste and increasing profits.

## Key Findings:
**Effective Categorization:** By fine-tuning open-source LLMs like Llama, BERT, and GPT-2, the project successfully categorized products into predefined hierarchical categories, such as food, beverages, and alcohol. This classification helps streamline inventory management processes.

**Model Fine-Tuning and Optimization:** The use of efficient fine-tuning techniques, particularly LoRA (low-rank adaptation), enabled enhanced performance with fewer computational resources. This approach led to improved accuracy and precision in product categorization.

**Insights from Misclassified Data:** Analysis revealed instances of mislabeling in the dataset sourced from Instacart. By identifying these inconsistencies, the models provided insights into areas where the dataset could be refined for better accuracy.

**Model Performance Comparison:** The study compared various models, demonstrating that even smaller models like BERT could achieve performance comparable to larger models like GPT-2, highlighting the potential for resource-efficient solutions without compromising accuracy.

## Conclusion and Recommendations
The project demonstrated the viability of using open-source LLMs for enhanced product categorization in the restaurant industry, offering an inexpensive and scalable solution for managing diverse inventory descriptions. We can also improve the performance of our models by expanding the dataset through scraping or data augmentation techniques. Future work will focus on integrating these insights into a personalized AI assistant for tailored restaurant business analytics, ultimately driving efficiency and reducing operational costs.

## Future Perspectives:
**Updating the Loss Function:** Develop a more sophisticated loss function to enhance the model's understanding of hierarchical relationships within categories.

**Expand Data Sources:** Increase the dataset size through additional scraping and data augmentation techniques to improve model robustness.

**AI Assistant Development:** Combine findings from inventory demand forecasting and categorization projects to create an AI assistant that provides real-time, actionable insights for restaurants, optimizing inventory and operational decisions.
