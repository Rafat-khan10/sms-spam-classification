import streamlit as st
import joblib


@st.cache_resource
def load_model():
    model=joblib.load("model.pkl")
    vectorizer=joblib.load("vectorizer.pkl")

    return model,vectorizer

model, vectorizer=load_model()

 
st.set_page_config( 
    page_title="Email/SMS Spam classification",
     layout="centered" )

st.markdown( """ 
            <h1 style="text-align:center;"> Email/SMS Spam classification</h1>
             <p style="text-align:center; color:gray;">Predict whether an Email/SMS is Spam or Ham using Machine Learning

            </p> """
            , 
            unsafe_allow_html=True )
st.divider()
st.subheader("🔍 Enter an Email/SMS to classify")
text = st.text_area( "", 
    placeholder="Paste an Email or SMS message here (offers, links, prizes etc.)"
,
      height=120 )


predict_btn = st.button("🚀 Predict", use_container_width=True)

if predict_btn:
    if text.strip() == "":
         st.warning("⚠️ Please enter a text first.")
    else:
        text_vector = vectorizer.transform([text]) 
        prediction = model.predict(text_vector)[0] 
        proba = model.predict_proba(text_vector)[0]

        ham_confidence = proba[0]*100
        spam_confidence = proba[1]*100

    
        st.divider() 

        if spam_confidence > 45:
             st.error("**🚨 Spam Message Detected!**")
             st.info(f"📊 Model is {spam_confidence:.2f}% confident that this message is SPAM")

        elif spam_confidence >35:
            st.warning("⚠️ Suspicious Message")

        else:
             st.success("**✅This is a Safe Message (Not Spam)**")
             st.info(f"📊 Model is {ham_confidence:.2f}% confident that this message is Safe")
  

st.caption("Built with ❤️ using Streamlit & Machine Learning")