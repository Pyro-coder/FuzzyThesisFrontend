from flask import Flask, render_template, request

app = Flask(__name__)

criteria = [
    {"name": "Glib", "score": "High", "description": "Very Glib"},
    {"name": "Grandiosity", "score": "High", "description": "Very Grandiose"},
    {"name": "Conning", "score": "High", "description": "Very Conniving"},
    {"name": "Pathological Lying", "score": "Moderate", "description": "Very Deceptive"},
    {"name": "Lack of Remorse", "score": "High", "description": "Very Unremorseful"},
    {"name": "Callousness", "score": "Moderate", "description": "Very Callous"},
    {"name": "Shallow Affect", "score": "Moderate", "description": "Very Inexpressive"},
    {"name": "Acceptance of Responsibilities of Actions", "score": "Low", "description": "Very Irresponsible"},
    {"name": "Need for Stimulation", "score": "Moderate", "description": "Mildly Restless"},
    {"name": "Realistic long term goals", "score": "Moderate", "description": "Very Practical"},
    {"name": "Impulsivity", "score": "Moderate", "description": "Very Impulsive"},
    {"name": "Irresponsibility", "score": "High", "description": "Very Irresponsible"},
    {"name": "Parasitic Lifestyle", "score": "High", "description": "Very Parasitic"},
    {"name": "Number of Short Term Marital Relationships", "score": "Very High", "description": "Very Noncommittal"},
    {"name": "Sexual Promiscuity", "score": "High", "description": "Fairly Promiscuous"},
    {"name": "Behavioral Control", "score": "High", "description": "Very Controlled"},
    {"name": "Early Behavioral Problems", "score": "High", "description": "Very Problematic"},
    {"name": "Juvenile Delinquency", "score": "High", "description": "Mildly Delinquent"},
    {"name": "Revocation of Conditional Release", "score": "Moderate", "description": "Fairly Noncompliant"},
    {"name": "Criminal Versatility", "score": "High", "description": "Very Versatile"}
]

scoring_criteria = {
    "Glib": ["Very Sincere", "Fairly Sincere", "Mildly Sincere", "Neutral", "Mildly Glib", "Fairly Glib", "Very Glib"],
    "Grandiosity": ["Very Humble", "Fairly Humble", "Mildly Humble", "Neutral", "Mildly Grandiose", "Fairly Grandiose", "Very Grandiose"],
    "Conning": ["Very Trusting", "Fairly Trusting", "Mildly Trusting", "Neutral", "Mildly Conniving", "Fairly Conniving", "Very Conniving"],
    "Pathological Lying": ["Very Truthful", "Fairly Truthful", "Mildly Truthful", "Neutral", "Mildly Deceptive", "Fairly Deceptive", "Very Deceptive"],
    "Lack of Remorse": ["Very Remorseful", "Fairly Remorseful", "Mildly Remorseful", "Neutral", "Mildly Unremorseful", "Fairly Unremorseful", "Very Unremorseful"],
    "Callousness": ["Very Caring", "Fairly Caring", "Mildly Caring", "Neutral", "Mildly Callous", "Fairly Callous", "Very Callous"],
    "Shallow Affect": ["Very Expressive", "Fairly Expressive", "Mildly Expressive", "Neutral", "Mildly Inexpressive", "Fairly Inexpressive", "Very Inexpressive"],
    "Acceptance of Responsibilities of Actions": ["Very Accountable", "Fairly Accountable", "Mildly Accountable", "Neutral", "Mildly Irresponsible", "Fairly Irresponsible", "Very Irresponsible"],
    "Need for Stimulation": ["Very Content", "Fairly Content", "Mildly Content", "Neutral", "Mildly Restless", "Fairly Restless", "Very Restless"],
    "Realistic long term goals": ["Very Unrealistic", "Fairly Unrealistic", "Mildly Unrealistic", "Neutral", "Mildly Practical", "Fairly Practical", "Very Practical"],
    "Impulsivity": ["Very Calculated", "Fairly Calculated", "Mildly Calculated", "Neutral", "Mildly Impulsive", "Fairly Impulsive", "Very Impulsive"],
    "Irresponsibility": ["Very Responsible", "Fairly Responsible", "Mildly Responsible", "Neutral", "Mildly Irresponsible", "Fairly Irresponsible", "Very Irresponsible"],
    "Parasitic Lifestyle": ["Very Independent", "Fairly Independent", "Mildly Independent", "Neutral", "Mildly Parasitic", "Fairly Parasitic", "Very Parasitic"],
    "Number of Short Term Marital Relationships": ["Very Committed", "Fairly Committed", "Mildly Committed", "Neutral", "Mildly Noncommittal", "Fairly Noncommittal", "Very Noncommittal"],
    "Sexual Promiscuity": ["Very Chaste", "Fairly Chaste", "Mildly Chaste", "Neutral", "Mildly Promiscuous", "Fairly Promiscuous", "Very Promiscuous"],
    "Behavioral Control": ["Very Uncontrolled", "Fairly Uncontrolled", "Mildly Uncontrolled", "Neutral", "Mildly Controlled", "Fairly Controlled", "Very Controlled"],
    "Early Behavioral Problems": ["Very Well-behaved", "Fairly Well-behaved", "Mildly Well-behaved", "Neutral", "Mildly Problematic", "Fairly Problematic", "Very Problematic"],
    "Juvenile Delinquency": ["Very Law-Abiding", "Fairly Law-Abiding", "Mildly Law-Abiding", "Neutral", "Mildly Delinquent", "Fairly Delinquent", "Very Delinquent"],
    "Revocation of Conditional Release": ["Very Compliant", "Fairly Compliant", "Mildly Compliant", "Neutral", "Mildly Noncompliant", "Fairly Noncompliant", "Very Noncompliant"],
    "Criminal Versatility": ["Very Specialized", "Fairly Specialized", "Mildly Specialized", "Neutral", "Mildly Versatile", "Fairly Versatile", "Very Versatile"]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission and scoring here
        results = request.form.to_dict()
        return render_template('results.html', results=results)
    return render_template('index.html', criteria=criteria, scoring_criteria=scoring_criteria)

if __name__ == '__main__':
    app.run(debug=True)
