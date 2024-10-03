# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable (optional, useful if you want the app not to open a browser window)
ENV STREAMLIT_SERVER_HEADLESS=true

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
