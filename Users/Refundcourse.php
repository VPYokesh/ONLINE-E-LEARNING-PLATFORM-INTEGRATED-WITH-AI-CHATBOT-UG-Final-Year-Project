<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancel Enrollment</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            margin: 0;
            padding: 20px;
        }
        .form-container {
            max-width: 500px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h2 {
            text-align: center;
            margin-bottom: 20px;
            color: #007bff;
        }
        form {
            margin-top: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        input[type="text"],
        textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            box-sizing: border-box;
        }
        textarea {
            height: 150px;
        }
        input[type="submit"] {
            background-color: #007bff;
            color: #fff;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        .back-button {
            display: block;
            text-align: center;
            margin-top: 20px;
        }
        .back-button a {
            text-decoration: none;
            color: #007bff;
            padding: 8px 16px;
            border: 1px solid #007bff;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .back-button a:hover {
            background-color: #007bff;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h2>Cancel Enrollment</h2>
        <?php
        // Start session
        include_once("../DB_Files/db.php");
        session_start();

        // Check if student is logged in
        if (isset($_SESSION['stu_id'])) {
            // Check if form is submitted
            if (isset($_POST['cancel'])) {
                // Fetch form data
                $stu_email = $_POST['stu_email'];
                $course_id = $_POST['course_id'];
                $reply = $_POST['course']; // User's reply from text area

                // Query to delete course order for the student
                $sql = "DELETE FROM courseorder WHERE stu_email='$stu_email' AND course_id='$course_id'";
                $result = mysqli_query($conn, $sql);

                if ($result) {
                    $sql_insert_reply = "INSERT INTO cancellation_responses (stu_email, course_id, reply) VALUES ('$stu_email', '$course_id', '$reply')";
                    $result_insert_reply = mysqli_query($conn, $sql_insert_reply);

                    if ($result_insert_reply) {
                        echo "Course enrollment cancelled successfully and reply saved.";
                    } else {
                        echo "Error saving reply: " . mysqli_error($conn);
                    }
                } else {
                    echo "Error cancelling enrollment: " . mysqli_error($conn);
                }
            }
        } else {
            echo "Error: Student is not logged in.";
        }
        ?>
        <div class="back-button">
            <a href="MyCourse.php">Back to Home</a>
        </div>
    </div>
</body>
</html>
