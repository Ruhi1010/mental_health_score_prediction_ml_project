(() => {
  "use strict";

  // Your actual Render backend URL
  const API_BASE =
    "https://mental-health-score-prediction-ml.onrender.com";

  const form = document.getElementById("predict-form");

  // Adjust these IDs if your HTML uses different IDs
  const resultBox = document.getElementById("result");
  const errorBox = document.getElementById("error");

  /* =========================================================
     COLLECT FORM DATA
     ========================================================= */

  function collectPayload() {
    const fd = new FormData(form);

    return {
      age:
        fd.get("age") === ""
          ? NaN
          : parseInt(fd.get("age"), 10),

      gender: fd.get("gender") || "",

      country: (fd.get("country") || "").trim(),

      academic_level: fd.get("academic_level") || "",

      most_used_platform:
        fd.get("most_used_platform") || "",

      purpose_of_use:
        fd.get("purpose_of_use") || "",

      avg_daily_usage_hours:
        fd.get("avg_daily_usage_hours") === ""
          ? NaN
          : parseFloat(fd.get("avg_daily_usage_hours")),

      daily_unlocks:
        fd.get("daily_unlocks") === ""
          ? NaN
          : parseInt(fd.get("daily_unlocks"), 10),

      study_hours:
        fd.get("study_hours") === ""
          ? NaN
          : parseFloat(fd.get("study_hours")),

      physical_activity_hours:
        fd.get("physical_activity_hours") === ""
          ? NaN
          : parseFloat(fd.get("physical_activity_hours")),

      sleep_hours_per_night:
        fd.get("sleep_hours_per_night") === ""
          ? NaN
          : parseFloat(fd.get("sleep_hours_per_night")),

      stress_level:
        fd.get("stress_level") || "",
    };
  }


  /* =========================================================
     CONVERT FRONTEND DATA TO FASTAPI DATA
     ========================================================= */

  function toApiPayload(payload) {
    return {
      Age: payload.age,
      Gender: payload.gender,
      Country: payload.country,
      Academic_Level: payload.academic_level,
      Most_Used_Platform: payload.most_used_platform,
      Purpose_Of_Use: payload.purpose_of_use,
      Avg_Daily_Usage_Hours: payload.avg_daily_usage_hours,
      Daily_Unlocks: payload.daily_unlocks,
      Study_Hours: payload.study_hours,
      Physical_Activity_Hours: payload.physical_activity_hours,
      Sleep_Hours_Per_Night: payload.sleep_hours_per_night,
      Stress_Level: payload.stress_level,
    };
  }


  /* =========================================================
     VALIDATION
     ========================================================= */

  function validate(payload) {
    const errors = [];

    if (
      !Number.isInteger(payload.age) ||
      payload.age < 10 ||
      payload.age > 100
    ) {
      errors.push("Age must be between 10 and 100.");
    }

    if (!payload.gender) {
      errors.push("Please select your gender.");
    }

    if (!payload.country) {
      errors.push("Please enter your country.");
    }

    if (!payload.academic_level) {
      errors.push("Please select your academic level.");
    }

    if (!payload.most_used_platform) {
      errors.push("Please select your most used platform.");
    }

    if (!payload.purpose_of_use) {
      errors.push("Please select your purpose of use.");
    }

    if (
      !Number.isFinite(payload.avg_daily_usage_hours) ||
      payload.avg_daily_usage_hours < 0 ||
      payload.avg_daily_usage_hours > 24
    ) {
      errors.push(
        "Average daily usage hours must be between 0 and 24."
      );
    }

    if (
      !Number.isInteger(payload.daily_unlocks) ||
      payload.daily_unlocks < 0
    ) {
      errors.push("Daily unlocks must be 0 or greater.");
    }

    if (
      !Number.isFinite(payload.study_hours) ||
      payload.study_hours < 0 ||
      payload.study_hours > 24
    ) {
      errors.push("Study hours must be between 0 and 24.");
    }

    if (
      !Number.isFinite(payload.physical_activity_hours) ||
      payload.physical_activity_hours < 0 ||
      payload.physical_activity_hours > 24
    ) {
      errors.push(
        "Physical activity hours must be between 0 and 24."
      );
    }

    if (
      !Number.isFinite(payload.sleep_hours_per_night) ||
      payload.sleep_hours_per_night < 0 ||
      payload.sleep_hours_per_night > 24
    ) {
      errors.push(
        "Sleep hours must be between 0 and 24."
      );
    }

    if (!payload.stress_level) {
      errors.push("Please select your stress level.");
    }

    return errors;
  }


  /* =========================================================
     DISPLAY ERROR
     ========================================================= */

  function showError(message) {
    if (errorBox) {
      errorBox.textContent = message;
      errorBox.style.display = "block";
    }

    console.error(message);
  }


  /* =========================================================
     CLEAR ERROR
     ========================================================= */

  function clearError() {
    if (errorBox) {
      errorBox.textContent = "";
      errorBox.style.display = "none";
    }
  }


  /* =========================================================
     DISPLAY RESULT
     ========================================================= */

  function showResult(score) {
    if (!resultBox) {
      console.warn(
        "Result element with id='result' was not found."
      );
      return;
    }

    resultBox.textContent =
      `Predicted Mental Health Score: ${score}`;

    resultBox.style.display = "block";
  }


  /* =========================================================
     FORM SUBMIT
     ========================================================= */

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    clearError();

    const payload = collectPayload();

    console.log("Frontend payload:", payload);

    /* Validate frontend data */
    const clientErrors = validate(payload);

    if (clientErrors.length > 0) {
      showError(clientErrors.join(" "));
      return;
    }

    /* Convert to FastAPI format */
    const apiPayload = toApiPayload(payload);

    console.log("API URL:", `${API_BASE}/predict`);
    console.log("API payload:", apiPayload);

    /* Disable button while predicting */
    const submitButton =
      form.querySelector('button[type="submit"]');

    const originalButtonText = submitButton
      ? submitButton.textContent
      : "";

    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Predicting...";
    }

    try {
      const response = await fetch(
        `${API_BASE}/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify(apiPayload),
        }
      );

      console.log(
        "API response status:",
        response.status
      );

      /* Get response body */
      const responseText = await response.text();

      console.log(
        "API raw response:",
        responseText
      );

      let data;

      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        throw new Error(
          "The server returned an invalid response."
        );
      }


      /* =====================================================
         SUCCESS
         ===================================================== */

      if (response.ok) {
        if (
          data.predicted_mental_health_score !== undefined
        ) {
          const score =
            data.predicted_mental_health_score;

          showResult(score);

          console.log(
            "Prediction successful:",
            score
          );
        } else {
          showError(
            "Prediction received, but the score was not found in the response."
          );
        }

        return;
      }


      /* =====================================================
         FASTAPI VALIDATION ERROR
         ===================================================== */

      if (response.status === 422) {
        console.error(
          "FastAPI validation error:",
          data
        );

        if (Array.isArray(data.detail)) {
          const messages = data.detail.map((error) => {
            const location =
              error.loc
                ? error.loc.join(" → ")
                : "Field";

            return `${location}: ${error.msg}`;
          });

          showError(
            "Validation error: " +
              messages.join(" | ")
          );
        } else {
          showError(
            "The submitted data is invalid."
          );
        }

        return;
      }


      /* =====================================================
         SERVER ERROR
         ===================================================== */

      if (response.status >= 500) {
        console.error(
          "Server error:",
          data
        );

        showError(
          "The prediction server encountered an error. Please try again."
        );

        return;
      }


      /* =====================================================
         OTHER HTTP ERROR
         ===================================================== */

      showError(
        data.detail ||
          data.message ||
          `Request failed with status ${response.status}.`
      );
    }

    catch (error) {
      console.error(
        "Prediction request failed:",
        error
      );

      showError(
        "Could not connect to the prediction server. Please check your internet connection or try again."
      );
    }

    finally {
      /* Re-enable button */
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent =
          originalButtonText || "Predict";
      }
    }
  });

})();