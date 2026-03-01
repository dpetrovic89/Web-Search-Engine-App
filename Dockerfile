# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy the rest of the application code into the container
COPY . .

# Create data and index directories if they don't exist
RUN mkdir -p data index/whoosh

# The port Hugging Face Spaces expects
ENV PORT=7860

# Run app.py when the container launches
CMD ["python", "app.py"]
