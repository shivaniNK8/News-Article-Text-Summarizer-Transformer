FROM continuumio/miniconda3

# Install python packages
RUN mkdir /opt/api
COPY requirements.txt /opt/api/
RUN pip install -r /opt/api/requirements.txt

# Copy files into container
COPY song-cluster-model.joblib /opt/api/
COPY spotify_app.py /opt/api/
COPY spotify_songs.csv /opt/api/
COPY .streamlit /opt/api/.streamlit
COPY *.jpeg /opt/api/

# Set work directory and open the required port
WORKDIR /opt/api
EXPOSE 8501

# Run our service script
CMD ["streamlit", "run","spotify_app.py"]
