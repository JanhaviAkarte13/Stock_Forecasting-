# 📈 Stock Forecasting Website

A full-stack **Flask web application** for **stock market forecasting** using machine learning models like ARIMA, Prophet, and Random Forest. The platform allows users to get stock predictions, analyze trends, visualize data, and interact with a Gemini-powered chatbot for stock insights.

---

## 🏗 Project Overview

This project combines **data science**, **machine learning**, and **web development** to provide an interactive platform for stock market enthusiasts and investors. Users can:

- Fetch and view stock data.
- Get predictions using multiple ML models.
- Visualize trends and indicators such as Moving Average (MA) and Relative Strength Index (RSI).
- Track trending stocks.
- Interact with a chatbot for market queries.

The backend is powered by **Flask**, ML models are in **Python**, and the frontend uses **HTML, CSS, and JavaScript** with **Chart.js** for visualizations.

---

## 🛠 Features

- **Stock Predictions:** Forecast stock prices using ARIMA, Prophet, and Random Forest models.  
- **Interactive Dashboard:** Monitor stock performance and key indicators.  
- **Trending Stocks:** See the latest top-performing stocks.  
- **Data Visualizations:** View charts and graphs for stock trends and analysis.  
- **Chatbot Integration:** Gemini-powered chatbot to answer stock-related queries.  
- **User Authentication:** Secure login and registration system.  
- **Responsive UI:** Works across desktop and mobile devices.  

---

## 🏛 Architecture

- **Flask Backend:** Handles routes, API calls, authentication, and rendering templates.  
- **ML Models:** Stored in `/models`, include preprocessing, feature engineering, forecasting, and evaluation.  
- **Database:** SQLite database for storing user info and stock data.  
- **Frontend:** HTML templates in `/templates` and static files in `/static` (CSS, JS, images).  
- **Utilities:** Custom decorators and helper functions in `/utils` for cleaner code.  

---

## 📋 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/stock-forecasting-website.git
   cd stock-forecasting-website
