const API_BASE = "http://localhost:8000/";
const PREDICT_URL = "predict/";

const validate = (fieldData) => {
    let errors = [];
    console.log(fieldData);
    for (const key in fieldData) {
        const value = fieldData[key];
        if (value === "") errors.push(`${key} is required`);
        else if (isNaN(value)) errors.push(`${key} must be a number`);

        switch (key) {
            case "age":
                if (value < 15 || value > 50)
                    errors.push("Age must be between 15 and 50");
                break;
            case "release_clause_eur":
            case "wage_eur":
            case "value_eur":
                break;
            default:
                if (value < 0 || value > 100)
                    errors.push(`${key} must be between 0 and 100`);
                break;
        }
    }
    return errors;
};

const validateAndReturnData = () => {
    const fieldIds = [
        "age",
        "movement_reactions",
        "wage_eur",
        "value_eur",
        "release_clause_eur",
        "mentality_composure",
        "physic",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
    ];

    let data = {};

    fieldIds.forEach((fieldId) => {
        data[fieldId] = Number.parseFloat(
            document.getElementById(fieldId).value.trim(),
        );
    });
    const errors = validate(data);

    if (errors.length > 0) {
        alert(errors.join("\n"));
        return null;
    }

    console.log(data);
    return data;
};

const onSubmit = async (e) => {
    e.preventDefault();
    const data = validateAndReturnData();
    data && predict(data);
};

const predict = async (data) => {
    const res = await fetch(`${API_BASE}${PREDICT_URL}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });

    const prediction = (await res.json()).data;
    document.getElementById("result").innerText = prediction;
    console.log(prediction);
};

const form = document.getElementById("form");
form.addEventListener("submit", onSubmit);
