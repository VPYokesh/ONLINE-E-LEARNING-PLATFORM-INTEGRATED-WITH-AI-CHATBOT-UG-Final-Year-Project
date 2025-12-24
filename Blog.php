<?php
include_once("Header.php");
include_once("DB_Files/db.php");
?>

<link rel="stylesheet" href="CSS/Blog.css">
<link rel="stylesheet" href="CSS/responsiveness.css">

<section class="reveal" id="blog-container">
    <div class="blogs">
        
        <?php
        $sql="SELECT * FROM blog";
        $result=$conn->query($sql);
        if($result->num_rows>0){
            while($row=$result->fetch_assoc()){
                $b_id=$row['b_id'];
                echo '
                <div class="post">
            <img src="'.str_replace('..','.',$row['b_img']).'" alt="">
            <br>
            <h3>'.$row['b_title'].' </h3>
            <br>
            <a href="BlogReadMore.php?b_id='.$b_id.'" class="button">Read More</a>
        </div>
        ';
            }
        }
        ?>
    </div>
</section>
<script>
    
function reveal() {
    var reveals = document.querySelectorAll(".reveal");

    for (var i = 0; i < reveals.length; i++) {
        var windowHeight = window.innerHeight;
        var elementTop = reveals[i].getBoundingClientRect().top;
        var elementVisible = 150;

        if (elementTop < windowHeight - elementVisible) {
            reveals[i].classList.add("active");
        } else {
            reveals[i].classList.remove("active");
        }
    }
}

window.addEventListener("scroll", reveal);
</script>
<!-- <iframe id="chatbot-frame" src="http://127.0.0.1:5000/" width="1900" height="1000" frameborder="0" style="position: fixed; bottom: 0px; right: 0px;"></iframe> -->
    <iframe allow="microphone" id="chatbot-frame" src="http://127.0.0.1:5000/" width="500" height="600" frameborder="0" style="position: fixed; bottom: 0px; right: 0px;"></iframe>
<?php
include_once("Footer.php");
?>