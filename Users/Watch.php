<?php
session_start();
include_once("../DB_Files/db.php");

if (!isset($_SESSION['stu_id'])) {
    header('Location:../Home.php');
    exit();
}

$link = $_REQUEST['link'];

// Perform the database query
$sql = "SELECT * FROM lesson WHERE lesson_link='$link'";
$result = $conn->query($sql);

// Check if the query was successful
if ($result) {
    // Fetch the row from the result set
    $row = $result->fetch_assoc();

    // Check if the row is not empty
    if ($row) {
        // Output the course and lesson information
        echo "<div class='container'>";
        echo "<div class='course-info bg-dark text-white py-3 mb-4'>";
        echo "<h4 class='text-center mb-0'>Course Name: " . $row['course_name'] . "</h4>";
        echo "</div>";
        echo "<div class='row'>";
        echo "<div class='col-md-8 mx-auto'>";
        echo "<div class='lesson-info bg-light p-4'>";
        echo "<h3 class='lesson-title mb-4'>Lesson Name: " . $row['lesson_name'] . "</h3>";
        echo "<div class='video-container mb-4'>";
        // Embed the YouTube video player here
        echo "<div id='player'></div>";
        echo "</div>";
        echo "<a href='WatchList.php?course_id=" . $row['course_id'] . "' class='btn btn-danger d-block mx-auto'>Finish</a>";
        echo "</div>"; // .lesson-info
        echo "</div>"; // .col-md-8
        echo "</div>"; // .row
        echo "</div>"; // .container
    } else {
        // Handle case where no rows were returned
        echo "No lesson found for the given link.";
    }
} else {
    // Handle case where the query failed
    echo "Error: " . mysqli_error($conn);
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-Learn</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
        }
        .course-info {
            border-radius: 8px;
        }
        .lesson-title {
            color: #333;
        }
        .video-container {
            position: relative;
            padding-top: 56.25%;
            overflow: hidden;
            margin-bottom: 20px;
        }
        #player {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        .btn-danger {
            background-color: #dc3545;
            border-color: #dc3545;
        }
    </style>
    <link rel="stylesheet" href="../bootstrap/css/bootstrap.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js" integrity="sha512-894YE6QWD5I59HgZOGReFYm4dnWc1Qt5NtvYSaNcOP+u1T9qYdvdihz0PPSiiqn/+/3e7Jo4EaG7TubfWGUrMQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
</head>
<body>

<script>
    // Load the YouTube Player API asynchronously
    var tag = document.createElement('script');
    tag.src = "https://www.youtube.com/iframe_api";
    var firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

    // Function to create the YouTube player
    var player;
    function onYouTubeIframeAPIReady() {
        player = new YT.Player('player', {
            height: '100%',
            width: '100%',
            videoId: '<?php echo $link; ?>',
            playerVars: {
                controls: 1, // Show player controls
                autoplay: 0, // Do not autoplay
                modestbranding: 1, // Prevent YouTube logo in control bar
                fs: 1 // Show full-screen button
            },
            events: {
                'onReady': onPlayerReady
            }
        });
    }

    // Function called when player is ready
    function onPlayerReady(event) {
        // Optional: Add event listeners or custom functions
    }
</script>
<iframe allow="microphone" id="chatbot-frame" src="http://127.0.0.1:5000/" width="500" height="600" frameborder="0" style="position: fixed; bottom: 0px; right: 0px;"></iframe>
</body>
</html>
