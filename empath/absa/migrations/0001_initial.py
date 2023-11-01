# Generated by Django 4.2.4 on 2023-11-01 07:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('crawler', '0010_navernewsarticle_keyword'),
    ]

    operations = [
        migrations.CreateModel(
            name='Analysis',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('keyword', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(max_length=20)),
                ('user_id', models.IntegerField(null=True)),
                ('num_sentence', models.IntegerField(null=True)),
                ('token_count', models.TextField(null=True)),
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='crawler.task')),
            ],
        ),
    ]