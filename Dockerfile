FROM python:3.7

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY benign.jpg .
COPY vggtraintop_kfold_model.h5 .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
