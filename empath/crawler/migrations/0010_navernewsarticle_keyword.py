# Generated by Django 4.2.4 on 2023-11-01 07:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0009_task_user_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='navernewsarticle',
            name='keyword',
            field=models.CharField(max_length=100, null=True),
        ),
    ]