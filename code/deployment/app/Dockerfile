# python version
FROM python:3.9

# working directory in container
WORKDIR /app

# copy current content to container
COPY . /app

# install streamlit
RUN pip install streamlit

# expose port 8501
EXPOSE 8501

# run in headless mode
ENV STREAMLIT_SERVER_HEADLESS=true

# run when container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]