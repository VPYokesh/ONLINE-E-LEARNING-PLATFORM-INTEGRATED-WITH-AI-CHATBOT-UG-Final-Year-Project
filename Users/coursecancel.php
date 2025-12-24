<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h2 {
            text-align: center;
        }
        form {
            margin-top: 20px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
        }
        input[type="text"],
        textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
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
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
<?php
include_once("../DB_Files/db.php");
session_start();

if (isset($_SESSION['stu_id'])) {
    $stu_email = $_SESSION["stu_email"];

    $sql = "SELECT course_id FROM courseorder WHERE stu_email='$stu_email'";
    $result = mysqli_query($conn, $sql);

    if ($result && mysqli_num_rows($result) > 0) {

        $row = mysqli_fetch_assoc($result);
        $course_id = $row['course_id'];
    } else {

        $course_id = "";
    }
} else {

    $stu_email = "";
    $course_id = "";
}
?>

<div class="container">
        <h2>Cancel Enrollment</h2>
        <form method="POST" action="Refundcourse.php">

            <input type="hidden" name="stu_email" value="<?php echo $stu_email; ?>">

            <input type="hidden" name="course_id" value="<?php echo $course_id; ?>">
            <label for="course">Reason for Cancellation:</label>
            <textarea name="course" id="course" placeholder="Please provide a reason for cancelling the enrollment..." rows="5"></textarea>
            <input type="submit" name="cancel" value="Cancel Enrollment">
        </form>
</div>
</body>
</html>





<!-- <form method="POST" action="Refundcourse.php">
    <input type="hidden" name="stu_email" value="<?php echo $stu_email; ?>">
    <input type="hidden" name="course_id" value="<?php echo $course_id; ?>">
    <textarea name="course" rows="5" cols="10"></textarea>
    <input type="submit" name="cancel" value="Cancel Enrollment">
</form> -->





































<!-- <form method="POST" action="Refundcourse.php">
    <input type="text" name="stu_email" value="student@example.com">
    <input type="text" name="course_id" value="123">
    <input type="submit" name="cancel" value="Cancel Enrollment">
</form> -->
